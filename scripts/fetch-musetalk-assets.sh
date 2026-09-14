#!/bin/bash
# Fetch the MuseTalk source tree and inference weights into a scratch workspace.
#
# Run this on a LUMI *login* node: compute nodes have no internet access.
# Downloads ~4 GB. Safe to re-run; the HuggingFace CLI skips files it already has.
#
#   export PROJECT=project_462000131
#   scripts/fetch-musetalk-assets.sh

set -euo pipefail

PROJECT="${PROJECT:-project_462000131}"
WORKDIR="${WORKDIR:-/scratch/$PROJECT/$USER/musetalk}"
SIF="${SIF:-/appl/local/laifs/containers/lumi-multitorch-latest.sif}"
export HF_HOME="${HF_HOME:-/scratch/$PROJECT/$USER/hf-cache}"

# Singularity mounts /scratch read-only by default. This is the same bind list that
# `module use /appl/local/laifs/modules; module load lumi-aif-singularity-bindings`
# exports; set it here so the script also works in a non-interactive shell, where Lmod
# is not initialised.
export SINGULARITY_BIND="${SINGULARITY_BIND:-/var/spool/slurmd,/pfs,/scratch,/projappl,/project,/flash,/appl,/boot}"

# `hf` comes from huggingface_hub, which the LAIF base image already ships. Note the
# v1 CLI takes file names positionally -- passing them to `--include` instead makes it
# silently download only one of them.
hf() { singularity exec "$SIF" hf "$@"; }

mkdir -p "$WORKDIR/logs" "$WORKDIR/runs" "$HF_HOME"
cd "$WORKDIR"

[ -d MuseTalk ] || git clone --depth 1 https://github.com/TMElyralab/MuseTalk.git
cd MuseTalk
mkdir -p models/musetalkV15 models/sd-vae models/whisper models/face-parse-bisent

# MuseTalk V1.5 U-Net. models/dwpose and models/syncnet are deliberately skipped:
# DWPose is replaced by MediaPipe here (see musetalk-preprocessing-mediapipe.py) and
# SyncNet is only used for training.
hf download TMElyralab/MuseTalk musetalkV15/musetalk.json musetalkV15/unet.pth \
    --local-dir models

hf download stabilityai/sd-vae-ft-mse config.json diffusion_pytorch_model.bin \
    --local-dir models/sd-vae

hf download openai/whisper-tiny config.json pytorch_model.bin preprocessor_config.json \
    --local-dir models/whisper

# Upstream download_weights.sh pulls the BiSeNet face-parsing weights from Google Drive
# with gdown, which rate-limits shared login-node IPs. These HuggingFace mirrors hold
# the same files.
hf download vivym/face-parsing-bisenet 79999_iter.pth resnet18-5c106cde.pth \
    --local-dir models/face-parse-bisent \
  || hf download camenduru/MuseTalk \
       face-parse-bisent/79999_iter.pth face-parse-bisent/resnet18-5c106cde.pth \
       --local-dir models

echo
echo "Weights under $WORKDIR/MuseTalk/models:"
find models -type f -not -path '*/.cache/*' -printf '%12s  %p\n' | sort -k2
