#!/bin/bash
#SBATCH --job-name=sglang-serve
#SBATCH --account=<project>
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=8
#SBATCH --mem=480G
#SBATCH --time=02:00:00

# Usage:
#   sbatch scripts/sglang-serve.sh
#
# Required environment variables (set before sbatch or export from your env):
#   SIF         - path to the built sglang-lumi.sif
#   MODEL       - HuggingFace model id or local path, e.g. meta-llama/Llama-3.1-8B-Instruct
#   HF_TOKEN    - HuggingFace token (if accessing a gated model)
#   PORT        - server port (default: 30000)
#
# Example:
#   export SIF=/scratch/$LUMI_PROJECT/$USER/lumi-sglang.sif
#   export MODEL=meta-llama/Llama-3.1-8B-Instruct
#   export HF_TOKEN=hf_...
#   sbatch scripts/sglang-serve.sh

set -euo pipefail

: "${SIF:?SIF must be set to the path of lumi-sglang.sif}"
: "${MODEL:?MODEL must be set to a HuggingFace model id or local path}"
PORT="${PORT:-30000}"

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

# MIOpen requires writable per-job cache dirs; it aborts if it cannot set
# permissions on its runtime files.
MIOPEN_DIR=$(mktemp -d)
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_DIR/cache
export MIOPEN_USER_DB=$MIOPEN_DIR/config

echo "Starting SGLang server on port $PORT with model $MODEL"
echo "Node: $(hostname)"

singularity exec \
    ${HF_TOKEN:+--env HF_TOKEN=$HF_TOKEN} \
    --env ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    "$SIF" \
    python -m sglang.launch_server \
        --model-path "$MODEL" \
        --host 0.0.0.0 \
        --port "$PORT" \
        --tensor-parallel-size 8 \
        --attention-backend triton \
        --disable-cuda-graph \
        --device cuda
