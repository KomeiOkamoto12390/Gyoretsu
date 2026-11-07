#!/usr/bin/env bash
# Set up a GPU (CUDA) development environment for Gyoretsu inside WSL2 Ubuntu.
#
# Run from inside WSL:   bash /mnt/c/Users/komei/Proj/Gyoretsu/scripts/setup_wsl_gpu.sh
#
# Do NOT install an NVIDIA driver inside WSL: the Windows driver is shared with WSL.
# No CUDA toolkit is needed either: jax[cuda12] pip wheels bundle CUDA and cuDNN.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${GYORETSU_VENV:-$HOME/.venvs/gyoretsu}"  # Linux filesystem: much faster than /mnt/c

echo "==> Checking GPU passthrough"
if ! command -v nvidia-smi >/dev/null; then
  echo "nvidia-smi not found. Update the Windows NVIDIA driver and run 'wsl --update'." >&2
  exit 1
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

echo "==> Installing Python tooling"
sudo apt-get update -y
sudo apt-get install -y python3 python3-venv python3-pip git

echo "==> Creating venv at $VENV"
python3 -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip
"$VENV/bin/python" -m pip install -e "$REPO[dev,gpu]"

echo "==> Disabling JAX GPU memory preallocation (the 6 GB card is shared with Windows)"
grep -q XLA_PYTHON_CLIENT_PREALLOCATE "$VENV/bin/activate" ||
  echo "export XLA_PYTHON_CLIENT_PREALLOCATE=false" >> "$VENV/bin/activate"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

echo "==> Verifying JAX sees the GPU"
"$VENV/bin/python" - <<'PY'
import jax
devs = jax.devices()
print("JAX", jax.__version__, devs)
assert any(d.platform == "gpu" for d in devs), "JAX did not find a CUDA device"
PY

echo "==> Running tests and the Gate 0 report on GPU"
cd "$REPO"
"$VENV/bin/python" -m pytest -q
"$VENV/bin/python" examples/gate0_mm1_gradients.py

echo
echo "Done. Activate with:  source $VENV/bin/activate"
