#!/bin/bash
# Install SGLang on LUMI using lumi-container-wrapper (pip-containerize).
#
# Run from a GPU allocation on LUMI-G, e.g.:
#   srun --account=project_462000131 --partition=standard-g \
#        --gpus-per-node=1 --time=01:00:00 --pty bash -l
#
# Then:
#   bash scripts/install-sglang-pip-containerize.sh
#
# Required env vars (with defaults):
#   PROJECT         CSC project number   (default: project_462000131)
#   SGLANG_VERSION  SGLang git tag       (default: v0.5.6.post2)
#   CONTAINER       LAIF .sif path       (default: pinned lumi-multitorch)

set -euo pipefail

PROJECT="${PROJECT:-project_462000131}"
SGLANG_VERSION="${SGLANG_VERSION:-v0.5.6.post2}"
CONTAINER="${CONTAINER:-/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260216_093549/lumi-multitorch-full-u24r64f21m43t29-20260216_093549.sif}"
INSTALL_PREFIX="/scratch/${PROJECT}/${USER}/sglang_env"

# ── 1. Cleanup ──────────────────────────────────────────────────────────────
echo "[1/7] Cleanup old attempts"
for p in "${INSTALL_PREFIX}" "${INSTALL_PREFIX}"_*; do
    [ -e "$p" ] && rm -rf "$p"
done
find "/tmp/${USER}" -maxdepth 1 -type d -name 'cw-*' -exec rm -rf {} + 2>/dev/null || true
find /tmp -maxdepth 1 -type f -user "${USER}" -name 'cw-tmp.*' -delete 2>/dev/null || true
rm -f /tmp/lumi_sglang.yaml /tmp/post_sglang.sh /tmp/empty_requirements.txt

# ── 2. Modules ──────────────────────────────────────────────────────────────
echo "[2/7] Load modules"
module purge
module load LUMI
module load cray-python
module load lumi-container-wrapper

# ── 3. Prevent host env leakage into the container ──────────────────────────
echo "[3/7] Isolate environment"
unset LD_LIBRARY_PATH PYTHONPATH SINGULARITY_BIND APPTAINER_BIND \
      SINGULARITYENV_LD_LIBRARY_PATH APPTAINERENV_LD_LIBRARY_PATH 2>/dev/null || true

# ── 4. Wrapper config: point at the pinned LAIF container ───────────────────
echo "[4/7] Create wrapper config"
WRAP_BIN="$(readlink -f "$(command -v pip-containerize)")"
WRAP_ROOT="$(dirname "$(dirname "${WRAP_BIN}")")"

cp "${WRAP_ROOT}/configs/lumi.yaml" /tmp/lumi_sglang.yaml
CONTAINER="${CONTAINER}" python3 - <<'PY'
import os, yaml
p = "/tmp/lumi_sglang.yaml"
with open(p) as f:
    cfg = yaml.safe_load(f)
cfg["defaults"]["container_src"] = os.environ["CONTAINER"]
cfg["defaults"]["composable"] = False
cfg["defaults"]["isolate"] = "yes"
with open(p, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print("Wrote", p)
PY

# ── 5. Post-install script ───────────────────────────────────────────────────
echo "[5/7] Write post-install script"
cat > /tmp/post_sglang.sh <<SH
#!/bin/bash
set -euo pipefail
source "\$CW_INSTALLATION_PATH/\$CW_ENV_NAME/bin/activate"

SRC_DIR="\$(mktemp -d /tmp/sglang-src.XXXXXX)"
trap 'rm -rf "\$SRC_DIR"' EXIT

git clone --depth 1 --branch ${SGLANG_VERSION} \
    https://github.com/sgl-project/sglang.git "\$SRC_DIR/sglang"

cd "\$SRC_DIR/sglang"

# Swap in the AMD pyproject before pip resolves deps.
# pyproject_other.toml removes all CUDA/NVIDIA-only deps (flashinfer,
# sgl-kernel, nvidia-cutlass-dsl, etc.) and aiter, which fails to build for
# gfx90a by default (ROCm/aiter#179).
# It defines the all_hip extra with ROCm-compatible packages only.
rm python/pyproject.toml
mv python/pyproject_other.toml python/pyproject.toml

# Install with all AMD/HIP extras.  Do NOT use --no-deps: the extras
# (petit_kernel, wave-lang, etc.) are needed at runtime, not just at build
# time.  Skipping them produces an install that imports but fails on inference.
python -m pip install --no-build-isolation "./python[all_hip]"

# Patch get_amdgpu_memory_capacity for MI250x/gfx90a.
# rocminfo output does not match SGLang's grep pattern on this architecture,
# producing float('') -> ValueError.  The except only catches FileNotFoundError,
# so the error crashes the server during argument parsing.  Add ValueError to
# the except and mirror the NVIDIA path's torch.cuda.mem_get_info() fallback.
python - <<'PY'
from pathlib import Path
import sglang.srt.utils.common as m

path = Path(m.__file__)
src = path.read_text()

old = '''    except FileNotFoundError:
        raise RuntimeError(
            "rocm-smi not found. Ensure AMD ROCm drivers are installed and accessible."
        )'''

new = '''    except (FileNotFoundError, ValueError):
        if torch.cuda.is_available():
            logger.warning(
                "Failed to get GPU memory capacity from rocminfo, "
                "falling back to torch.cuda.mem_get_info()."
            )
            return torch.cuda.mem_get_info()[1] // 1024 // 1024  # unit: MB
        raise RuntimeError("Cannot determine AMD GPU memory capacity.")'''

assert old in src, f"Patch target not found in {path} — check SGLang version"
path.write_text(src.replace(old, new, 1))
print(f"Patched {path}")
PY

# pip check is intentionally omitted: SGLang and the container's vLLM have
# conflicting dep versions (grpcio, openai, outlines-core, etc.). The
# conflicts are in vLLM's side of the environment; SGLang itself installs
# correctly and the two engines are used independently.
SH
chmod +x /tmp/post_sglang.sh

# ── 6. Build the pip-containerize env ───────────────────────────────────────
echo "[6/7] Build container env (this takes ~10-20 min)"
: > /tmp/empty_requirements.txt
CW_GLOBAL_YAML=/tmp/lumi_sglang.yaml \
pip-containerize new \
    --system-site-packages \
    --prefix "${INSTALL_PREFIX}" \
    --post-install /tmp/post_sglang.sh \
    /tmp/empty_requirements.txt

# ── 7. Verify ───────────────────────────────────────────────────────────────
echo "[7/7] Verify"
PY="${INSTALL_PREFIX}/bin/python"

"$PY" - <<'PY'
import importlib, torch

for name in ("sglang", "torch", "triton", "petit_kernel"):
    try:
        m = importlib.import_module(name)
        print(f"{name}: {getattr(m, '__version__', getattr(m, '__file__', 'ok'))}")
    except ImportError as e:
        print(f"WARNING: {name} not importable: {e}")

print(f"ROCm PyTorch: {'+rocm' in torch.__version__}")
print(f"GPU count: {torch.cuda.device_count()}")
PY

# Confirm the server entrypoint loads without errors
SGLANG_USE_AITER=0 "$PY" -m sglang.launch_server --help > /dev/null

echo
echo "=========================================="
echo "SUCCESS"
echo "=========================================="
echo "To start the server:"
echo ""
echo "  SGLANG_USE_AITER=0 \\"
echo "  ${INSTALL_PREFIX}/bin/python -m sglang.launch_server \\"
echo "      --model-path /scratch/${PROJECT}/${USER}/models/<model-name> \\"
echo "      --host 127.0.0.1 \\"
echo "      --port 8000 \\"
echo "      --tp-size 1 \\"
echo "      --attention-backend triton"
