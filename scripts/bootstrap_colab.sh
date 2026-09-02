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

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: NVIDIA GPU runtime not found. Enable a Colab GPU first." >&2
    exit 1
fi

ensure_uv

if [[ ! -f uv.lock ]]; then
    cat >&2 <<'MSG'
ERROR: uv.lock is missing.
Generate and commit it from Windows with:

    powershell -ExecutionPolicy Bypass -File scripts/update_lock.ps1
MSG
    exit 2
fi

# Cloud runs must never silently re-resolve dependencies.
uv lock --check
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
uv sync --locked --group dev

uv run --locked python - <<'PY'
import torch
import peft
import transformers
import trl

print(f"torch={torch.__version__} cuda={torch.version.cuda}")
print(f"transformers={transformers.__version__}")
print(f"trl={trl.__version__} peft={peft.__version__}")

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available inside the locked project environment.")

for i in range(torch.cuda.device_count()):
    name = torch.cuda.get_device_name(i)
    major, minor = torch.cuda.get_device_capability(i)
    total_gib = torch.cuda.get_device_properties(i).total_memory / (1024**3)
    print(f"gpu[{i}]={name} sm={major}{minor} vram={total_gib:.1f}GiB")
PY

# Downloads are explicit here; ordinary training and evaluation remain local-only.
uv run --locked posttrain-math model download
uv run --locked posttrain-math data download
uv run --locked posttrain-math data prepare

uv run --locked pytest -q
uv run --locked posttrain-math environment

uv run --locked posttrain-math eval \
    --split dev \
    --prompt boxed \
    --limit "$SMOKE_LIMIT" \
    --batch-size 1 \
    --max-new-tokens 64 \
    --output-dir "$SMOKE_OUTPUT"

echo
echo "Colab bootstrap complete."
echo "- locked env:  .venv/"
echo "- model:       models/qwen3-1.7b-base/"
echo "- data:        data/raw/ and data/processed/"
echo "- smoke eval:  $SMOKE_OUTPUT"
echo "- run output:  runs/ (or pass a Google Drive path with --output-dir)"
