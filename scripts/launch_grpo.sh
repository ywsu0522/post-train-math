#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/launch_grpo.sh <1|2|auto> [GRPO args...]" >&2
  exit 2
fi

requested="$1"
shift

if [[ "$requested" == "auto" ]]; then
  requested="$(
    uv run --locked python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
  )"
fi

visible="$(
  uv run --locked python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)"

if ! [[ "$requested" =~ ^[0-9]+$ ]] || [[ "$requested" -lt 1 ]]; then
  echo "GPU count must be 1, 2, ... or auto; got: $requested" >&2
  exit 2
fi

if [[ "$requested" -gt "$visible" ]]; then
  echo "Requested $requested GPU(s), but only $visible are visible." >&2
  exit 1
fi

if [[ "$requested" -eq 1 ]]; then
  exec uv run --locked python -m posttrain_math.rl "$@"
fi

exec uv run --locked torchrun \
  --standalone \
  --nproc_per_node="$requested" \
  --module posttrain_math.rl \
  "$@"
