#!/bin/bash
#SBATCH --job-name=musetalk-inference
#SBATCH --account=project_462000131
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm-musetalk-%j.out

# MuseTalk V1.5 inference on one MI250x GCD.
#
# Usage, from the workspace that scripts/fetch-musetalk-assets.sh created:
#   cd /scratch/project_462000131/$USER/musetalk
#   sbatch --chdir=$PWD path/to/musetalk-inference.sh
#
# Override any of the variables below by exporting them before sbatch; the job
# inherits the calling environment.
#   CONTAINER  - SIF to run in (default: the plain LAIF base; point it at the built
#                musetalk-lumi.sif once you have one)
#   VENV       - Python environment holding diffusers/mediapipe/librosa/omegaconf
#                (default: the scratch venv; /opt/musetalk-venv in the built container)
#   CONFIG     - MuseTalk inference config (default: configs/inference/test.yaml)
#
# An hour of wall time is for the first run, which compiles MIOpen kernels. Once
# miopen-cache/ is warm the same job finishes in a few minutes.

set -euo pipefail

WORKDIR="${WORKDIR:-$(pwd)}"
CONTAINER="${CONTAINER:-/appl/local/laifs/containers/lumi-multitorch-latest.sif}"
VENV="${VENV:-$WORKDIR/venv}"
CONFIG="${CONFIG:-configs/inference/test.yaml}"
RESULT_DIR="${RESULT_DIR:-$WORKDIR/results}"

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

# MIOpen aborts if it cannot write its kernel cache, and $HOME is not a good place for
# it on a shared filesystem. Keep this directory across jobs: the LAIF image ships
# MIOpen's gfx90a *perf* databases but no precompiled kernel database (no .kdb, unlike
# LUMI's own lumi-pytorch-rocm images), so every convolution shape is compiled from
# source on first use. S3FD alone has 31 convolutions over a full video frame, which is
# a one-off cost of roughly ten minutes -- paid once, then served from here.
MIOPEN_DIR="$WORKDIR/miopen-cache"
mkdir -p "$MIOPEN_DIR" "$RESULT_DIR"
export SINGULARITYENV_MIOPEN_USER_DB_PATH="$MIOPEN_DIR"
export SINGULARITYENV_MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_DIR"

# S3FD downloads its weights from the internet on first use, which compute nodes cannot
# reach. fetch-musetalk-assets.sh primes this cache from a login node.
export SINGULARITYENV_TORCH_HOME="${TORCH_HOME:-$WORKDIR/torch-home}"
export SINGULARITYENV_HF_HOME="${HF_HOME:-$(dirname "$WORKDIR")/hf-cache}"
export SINGULARITYENV_HF_HUB_OFFLINE=1

export SINGULARITYENV_PYTHONNOUSERSITE=1
export SINGULARITYENV_PYTHONUNBUFFERED=1
export SINGULARITYENV_TOKENIZERS_PARALLELISM=false

# Only needed when running against the plain LAIF base: MediaPipe's native library links
# libEGL/libGLESv2, which the base image does not ship. musetalk-lumi.def installs them,
# so this is empty for the built container.
export SINGULARITYENV_MUSETALK_EXTRA_LIBS="${MUSETALK_EXTRA_LIBS:-}"

echo "node:      $(hostname)"
echo "container: $CONTAINER"
echo "venv:      $VENV"
echo "config:    $CONFIG"

srun singularity exec "$CONTAINER" bash -c "
set -euo pipefail
export LD_LIBRARY_PATH=\${MUSETALK_EXTRA_LIBS:+\$MUSETALK_EXTRA_LIBS:}\${LD_LIBRARY_PATH:-}

# The LAIF base puts its own /opt/venv on PYTHONPATH from
# /.singularity.d/env/10-docker2singularity.sh, which is sourced before this runs. That
# entry precedes our site-packages, so prepend or the base copies win.
VENV_SITE=\$('$VENV/bin/python' -c 'import site; print(site.getsitepackages()[0])')
export PYTHONPATH=\"\$VENV_SITE\${PYTHONPATH:+:\$PYTHONPATH}\"

cd '$WORKDIR/MuseTalk'
exec '$VENV/bin/python' -m scripts.inference \
    --inference_config '$CONFIG' \
    --result_dir '$RESULT_DIR' \
    --unet_model_path models/musetalkV15/unet.pth \
    --unet_config models/musetalkV15/musetalk.json \
    --whisper_dir models/whisper \
    --vae_type sd-vae \
    --version v15 \
    --gpu_id 0 \
    --ffmpeg_path /usr/bin
"
