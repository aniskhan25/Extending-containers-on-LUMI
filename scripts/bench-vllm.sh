#!/bin/bash
#SBATCH --job-name=bench-vllm
#SBATCH --account=<project>
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=1
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=logs/bench-vllm-%j.out

# Sweep vLLM serving throughput across concurrency levels on one MI250x GCD,
# for comparison with scripts/bench-sglang.sh.
#
# Usage:
#   sbatch --account=$PROJECT --chdir=$PWD scripts/bench-vllm.sh
#
# vLLM runs from the LAIF BASE image, not the SGLang container: the container's
# %environment prepends its venv to PYTHONPATH, which gives transformers 4.57.1,
# while vLLM 0.22.1 requires >= 5.9. The base ships vllm built for gfx90a
# (0.22.1+lumi.aif.gfx90a), so this compares vLLM as LUMI provides it against
# SGLang as we had to patch it.
#
# The benchmark CLIENT still runs from the SGLang image, so neither engine is
# measured by its own tooling.
#
#   EAGER=1   also run a second pass with --enforce-eager, which isolates how
#             much of any gap comes from CUDA graphs (SGLang cannot use them on
#             gfx90a) rather than from kernels.

set -euo pipefail

# SLURM copies the batch script to a spool directory, so $0 does not point at
# the repo. Resolve the helper against the submit directory instead.
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    BENCH_DIR="$SLURM_SUBMIT_DIR/scripts"
else
    BENCH_DIR="$(dirname "$0")"
fi
source "$BENCH_DIR/bench-common.sh"

LAIF_BASE="${LAIF_BASE:-/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260807_115122/lumi-multitorch-full-u24r70f21m50t210-20260807_115122.sif}"
OUTDIR="${OUTDIR:-/scratch/${PROJECT}/${USER}/sglang/bench/vllm}"
PORT="${PORT:-8000}"
EAGER="${EAGER:-0}"

: "${MODEL:?MODEL must point at a local model directory}"

mkdir -p "$OUTDIR"
load_modules

echo "=== node $(hostname)  $(date) ==="
echo "server image  $LAIF_BASE"
echo "client image  $SIF"
echo "model         $MODEL"
echo "tp            $TP"
echo "sweep         $CONCURRENCY  ($NUM_PROMPTS prompts, ${INPUT_LEN} in / ${OUTPUT_LEN} out)"

# serve_and_sweep <outdir> <server_log> [extra vllm flags...]
serve_and_sweep() {
    local outdir=$1 server_log=$2
    shift 2

    mkdir -p "$outdir"
    singularity exec "$LAIF_BASE" \
        vllm serve "$MODEL" \
            --served-model-name default \
            --host 127.0.0.1 \
            --port "$PORT" \
            --tensor-parallel-size "$TP" \
            --max-model-len 8192 \
            "$@" \
        > "$server_log" 2>&1 &
    local srv=$!

    if ! wait_for_health "$PORT" "$srv" 900; then
        echo "=== vLLM server failed to start ($*) ==="
        tail -80 "$server_log"
        kill $srv 2>/dev/null
        return 1
    fi

    run_sweep vllm "$PORT" "$outdir"

    kill $srv 2>/dev/null
    wait $srv 2>/dev/null
    return 0
}

serve_and_sweep "$OUTDIR" "$OUTDIR/server-${SLURM_JOB_ID:-manual}.log"

if [ "$EAGER" = "1" ]; then
    echo "=== second pass: --enforce-eager ==="
    serve_and_sweep "$OUTDIR-eager" "$OUTDIR-eager/server-${SLURM_JOB_ID:-manual}.log" --enforce-eager
fi

echo "=== done: $(ls "$OUTDIR"/c*.jsonl 2>/dev/null | wc -l) point(s) in $OUTDIR ==="
