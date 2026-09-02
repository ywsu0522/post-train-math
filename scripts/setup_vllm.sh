#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VLLM_VERSION="${VLLM_VERSION:-0.27.1}"
VLLM_ENV="${VLLM_ENV:-.venv-vllm}"

if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv is required. Run scripts/bootstrap_colab.sh first." >&2
    exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: NVIDIA GPU runtime not found." >&2
    exit 1
fi

# vLLM stays separate because its PyTorch/CUDA wheel requirements are tightly coupled.
uv venv "$VLLM_ENV" --python 3.12 --managed-python
uv pip install \
    --python "$VLLM_ENV/bin/python" \
    "vllm==$VLLM_VERSION" \
    --torch-backend=auto

"$VLLM_ENV/bin/python" - <<'PY'
import torch
import vllm

print(f"vllm={vllm.__version__}")
print(f"torch={torch.__version__} cuda={torch.version.cuda}")
print(f"gpu={torch.cuda.get_device_name(0)}")
print(f"sm={torch.cuda.get_device_capability(0)}")
PY

echo
echo "vLLM environment ready: $VLLM_ENV/"
