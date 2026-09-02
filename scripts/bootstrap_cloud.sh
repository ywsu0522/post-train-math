#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

UV_MIN_VERSION="0.12.6"
SMOKE_LIMIT="${POSTTRAIN_SMOKE_LIMIT:-2}"
SMOKE_OUTPUT="${POSTTRAIN_SMOKE_OUTPUT:-/tmp/posttrain-math-bootstrap-smoke}"

version_ge() {
    python - "$1" "$2" <<'PY'
import sys

def v(s: str) -> tuple[int, ...]:
    core = s.split("+", 1)[0].split("-", 1)[0]
    return tuple(int(part) for part in core.split(".") if part.isdigit())

raise SystemExit(0 if v(sys.argv[1]) >= v(sys.argv[2]) else 1)
PY
}

ensure_uv() {
    if command -v uv >/dev/null 2>&1; then
        current="$(uv --version | awk '{print $2}')"
        if version_ge "$current" "$UV_MIN_VERSION"; then
            echo "uv: $current"
            return
        fi
        echo "Updating uv ($current -> >=$UV_MIN_VERSION)..."
    else
        echo "Installing uv >=$UV_MIN_VERSION..."
    fi

    python -m pip install -q -U "uv>=$UV_MIN_VERSION"
    echo "uv: $(uv --version | awk '{print $2}')"
}

ensure_venv() {
    if [[ -x .venv/bin/python ]]; then
        current_py="$(.venv/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
        if [[ "$current_py" == "3.12" ]]; then
            echo "venv: .venv (Python $current_py)"
            return
        fi
        echo "Replacing .venv (Python $current_py -> 3.12)..."
        rm -rf .venv
    fi

    uv venv .venv --python 3.12 --managed-python
}

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: NVIDIA GPU runtime not found. Enable a GPU accelerator first." >&2
    exit 1
fi

ensure_uv
ensure_venv

nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader

# One install resolves the repo, dev tools, vLLM, and bitsandbytes.
# [tool.uv.pip] in pyproject.toml selects the PyTorch backend from the GPU driver.
uv pip install \
    --python .venv/bin/python \
    --editable . \
    --all-extras \
    --group dev

.venv/bin/python - <<'PY'
import torch
import bitsandbytes
import peft
import transformers
import trl
import vllm

print(f"torch={torch.__version__} cuda={torch.version.cuda}")
print(f"transformers={transformers.__version__}")
print(f"trl={trl.__version__} peft={peft.__version__}")
print(f"vllm={vllm.__version__} bitsandbytes={bitsandbytes.__version__}")

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available inside .venv.")

for i in range(torch.cuda.device_count()):
    name = torch.cuda.get_device_name(i)
    major, minor = torch.cuda.get_device_capability(i)
    total_gib = torch.cuda.get_device_properties(i).total_memory / (1024**3)
    print(f"gpu[{i}]={name} sm={major}{minor} vram={total_gib:.1f}GiB")

major, minor = torch.cuda.get_device_capability(0)
if (major, minor) < (7, 5):
    raise SystemExit(
        f"GPU 0 has compute capability {major}.{minor}; "
        "this cloud stack requires >= 7.5 for vLLM."
    )
PY

# Explicit bootstrap downloads; ordinary training/evaluation stays local-only.
.venv/bin/postrain-math model download
.venv/bin/postrain-math data download
.venv/bin/postrain-math data prepare

.venv/bin/pytest -q
.venv/bin/postrain-math environment

rm -rf "$SMOKE_OUTPUT"
.venv/bin/postrain-math eval \
    --split dev \
    --prompt boxed \
    --limit "$SMOKE_LIMIT" \
    --batch-size 1 \
    --max-new-tokens 64 \
    --output-dir "$SMOKE_OUTPUT"

echo
echo "Cloud bootstrap complete."
echo "- environment: .venv/"
echo "- model:       models/qwen3-1.7b-base/"
echo "- data:        data/raw/ and data/processed/"
echo "- smoke eval:  $SMOKE_OUTPUT"
