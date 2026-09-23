#!/bin/bash
#SBATCH --job-name=bench-sglang
#SBATCH --account=<project>
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=1
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=logs/bench-sglang-%j.out

# Sweep SGLang serving throughput across concurrency levels on one MI250x GCD.
#
# Usage:
#   sbatch --account=$PROJECT --chdir=$PWD scripts/bench-sglang.sh
#
# Configuration lives in scripts/bench-common.sh; override by exporting before
# sbatch. Results land in $OUTDIR as one c<N>.jsonl per concurrency level, then
# scripts/bench-report.py turns both engines' output into the README table.
#
# Run scripts/bench-vllm.sh for the other half of the comparison.

set -euo pipefail

# SLURM copies the batch script to a spool directory, so $0 does not point at
# the repo. Resolve the helper against the submit directory instead.
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    BENCH_DIR="$SLURM_SUBMIT_DIR/scripts"
else
    BENCH_DIR="$(dirname "$0")"
fi
source "$BENCH_DIR/bench-common.sh"

OUTDIR="${OUTDIR:-/scratch/${PROJECT}/${USER}/sglang/bench/sglang}"
PORT="${PORT:-30000}"
SERVER_LOG="${SERVER_LOG:-$OUTDIR/server-${SLURM_JOB_ID:-manual}.log}"

: "${MODEL:?MODEL must point at a local model directory}"

mkdir -p "$OUTDIR"
load_modules

echo "=== node $(hostname)  $(date) ==="
echo "image  $SIF"
echo "model  $MODEL"
echo "tp     $TP"
echo "sweep  $CONCURRENCY  ($NUM_PROMPTS prompts, ${INPUT_LEN} in / ${OUTPUT_LEN} out)"

# --attention-backend triton and --disable-cuda-graph are required on gfx90a;
# see the container notes in README.org. --served-model-name matches what the
# vLLM job serves, so the client sends the same model field to both.
# Multi-GCD runs need every GCD visible to the container; at tp=1 the allocation
# already scopes it, and listing all eight would be wrong.
ROCR_ARG=()
[ "$TP" -gt 1 ] && ROCR_ARG=(--env ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7)

singularity exec "${ROCR_ARG[@]}" "$SIF" \
    python -m sglang.launch_server \
        --model-path "$MODEL" \
        --served-model-name default \
        --host 127.0.0.1 \
        --port "$PORT" \
        --tp-size "$TP" \
        --attention-backend triton \
        --disable-cuda-graph \
    > "$SERVER_LOG" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null || true' EXIT

if ! wait_for_health "$PORT" "$SRV" 600; then
    echo "=== SGLang server failed to start ==="
    tail -80 "$SERVER_LOG"
    exit 1
fi

run_sweep sglang-oai "$PORT" "$OUTDIR"

# `wait` reports the signal status of the server we just killed, which would
# trip `set -e` and fail the job after the sweep already succeeded.
kill $SRV 2>/dev/null || true
wait $SRV 2>/dev/null || true

echo "=== done: $(ls "$OUTDIR"/c*.jsonl 2>/dev/null | wc -l) point(s) in $OUTDIR ==="
