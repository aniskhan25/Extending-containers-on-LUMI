#!/bin/bash
# Shared helpers for the SGLang vs vLLM serving benchmark.
#
# Sourced by scripts/bench-sglang.sh and scripts/bench-vllm.sh; not run directly.
#
# Override any of these by exporting them before sbatch; the job inherits the
# calling environment.
#
#   PROJECT       CSC project number            (default: project_462000131)
#   MODEL         model path                    (default: Llama-3.1-8B-Instruct on scratch)
#   SIF           SGLang image, used as the benchmark CLIENT for both engines
#   CONCURRENCY   levels to sweep               (default: 1 8 32 64)
#   NUM_PROMPTS   max prompts per point         (default: 600; scaled down at low concurrency)
#   INPUT_LEN     synthetic prompt tokens       (default: 1024)
#   OUTPUT_LEN    generated tokens per request  (default: 256)
#   SEED          dataset seed                  (default: 42)
#   TP            tensor-parallel size          (default: 1; needs matching --gpus-per-node)

PROJECT="${PROJECT:-project_462000131}"
MODEL="${MODEL:-/scratch/${PROJECT}/${USER}/models/Llama-3.1-8B-Instruct}"
SIF="${SIF:-/scratch/${PROJECT}/${USER}/containers/sglang-lumi.sif}"
CONCURRENCY="${CONCURRENCY:-1 8 32 64}"
NUM_PROMPTS="${NUM_PROMPTS:-600}"
INPUT_LEN="${INPUT_LEN:-1024}"
OUTPUT_LEN="${OUTPUT_LEN:-256}"
SEED="${SEED:-42}"
TP="${TP:-1}"

load_modules() {
    module purge
    module use /appl/local/laifs/modules
    module load lumi-aif-singularity-bindings
}

# wait_for_health <port> <server_pid> <timeout_s>
# Fails fast if the server process dies, so a crash costs seconds rather than
# the whole timeout.
wait_for_health() {
    local port=$1 pid=$2 timeout=$3
    local waited=0
    while [ "$waited" -lt "$timeout" ]; do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "server pid $pid died after ${waited}s"
            return 1
        fi
        if curl -sf -m 3 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
            echo "server ready after ${waited}s"
            return 0
        fi
        sleep 5
        waited=$((waited + 5))
    done
    echo "server not ready within ${timeout}s"
    return 1
}

# bench_once <backend> <port> <concurrency> <out_json>
#
# Both engines are driven through /v1/completions. bench_serving maps the
# sglang-oai and vllm backends to the same async_request_openai_completions
# function, so the client code path is identical. Do NOT use --backend sglang:
# it hits SGLang's native /generate and would compare two different paths.
#
# The client always runs from the SGLang image, including when benchmarking
# vLLM, so neither engine is measured by its own tooling.
bench_once() {
    local backend=$1 port=$2 conc=$3 out=$4
    singularity exec "$SIF" \
        python -m sglang.bench_serving \
            --backend "$backend" \
            --host 127.0.0.1 --port "$port" \
            --model default \
            --tokenizer "$MODEL" \
            --dataset-name random \
            --random-input-len "$INPUT_LEN" \
            --random-output-len "$OUTPUT_LEN" \
            --num-prompts "$NUM_PROMPTS" \
            --max-concurrency "$conc" \
            --warmup-requests 20 \
            --seed "$SEED" \
            --tag "tp$TP" \
            --disable-tqdm \
            --output-file "$out"
}

# run_sweep <backend> <port> <outdir>
run_sweep() {
    local backend=$1 port=$2 outdir=$3
    mkdir -p "$outdir"

    # One discarded pass absorbs Triton JIT, which would otherwise land entirely
    # in the first measured point. --warmup-requests handles per-point warmup.
    echo "=== JIT warmup (discarded) ==="
    local scratch
    scratch=$(mktemp)
    NUM_PROMPTS=32 bench_once "$backend" "$port" 8 "$scratch" || true
    rm -f "$scratch"

    for c in $CONCURRENCY; do
        # Scale prompts with concurrency so every point takes a comparable wall
        # time. At concurrency 1 the full NUM_PROMPTS would run serially and a
        # single point could outlast the job; throughput is a rate, so fewer
        # prompts there costs precision, not validity.
        local n=$((c * 20))
        [ "$n" -lt 40 ] && n=40
        [ "$n" -gt "$NUM_PROMPTS" ] && n="$NUM_PROMPTS"

        echo "=== concurrency $c  ($n prompts) ==="
        if ! NUM_PROMPTS=$n bench_once "$backend" "$port" "$c" "$outdir/c$c.jsonl"; then
            # Record nothing and keep going: a level that OOMs or times out is a
            # result, and the remaining levels are still worth measuring.
            echo "concurrency $c FAILED"
        fi
    done
}
