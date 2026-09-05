#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: bash scripts/eval_checkpoints.sh <1|2|auto> <training-run-dir> [eval-output-root]" >&2
  exit 2
fi

gpus="$1"
run_dir="${2%/}"
output_root="${3:-runs/evals/$(basename "$run_dir")}"
batch_size="${BATCH_SIZE:-2}"
max_new_tokens="${MAX_NEW_TOKENS:-1024}"

if [[ ! -d "$run_dir" ]]; then
  echo "Training run directory not found: $run_dir" >&2
  exit 1
fi

mkdir -p "$output_root"

mapfile -t checkpoints < <(
  find "$run_dir" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V
)

models=("${checkpoints[@]}")
if [[ -d "$run_dir/final-model" ]]; then
  models+=("$run_dir/final-model")
fi

if [[ "${#models[@]}" -eq 0 ]]; then
  echo "No checkpoint-* or final-model found under $run_dir" >&2
  exit 1
fi

for model in "${models[@]}"; do
  name="$(basename "$model")"
  echo
  echo "=== Evaluating $name ==="
  bash scripts/launch_gpu.sh "$gpus" \
    eval \
    --model "$model" \
    --split dev \
    --prompt boxed \
    --batch-size "$batch_size" \
    --max-new-tokens "$max_new_tokens" \
    --output-dir "$output_root/$name"
done

uv run --locked python scripts/summarize_eval_runs.py "$output_root"
