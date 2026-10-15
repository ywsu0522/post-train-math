# OLMo2-1B: fixed boxed-1shot / L1–3 comparison

This is an explicit **research contract**, not a claim that official SFT is free of MATH leakage. All commands assume the working directory is the repository root and a compatible locked Python/CUDA environment is installed.

## Source pins

| Item | Identifier | Immutable revision |
| --- | --- | --- |
| Official full SFT | `allenai/OLMo-2-0425-1B-SFT` | `0d85a3d037876ce6ac7d4311d994400fc66ac27f` |
| Original base | `allenai/OLMo-2-0425-1B` | `13cece9360d59bb9db636273ea8d000b67fcc27b` |
| MATH | `DigitalLearningGmbH/MATH-lighteval` | `f06834690385b29df31ccc717250746a3ba0322b` |

`OLMo-2-0425-1B-SFT` is **full weights**, not a LoRA adapter. It has its own tokenizer and a chat template. In this comparison we intentionally select the same **plain-text** `boxed-1shot` prompt on every model because we want one fixed prompt contract; this is **not** the model's native `apply_chat_template()` usage and may reduce performance. Do not combine the chat template and this raw prompt without declaring a different experiment.

The official Tülu 3 SFT mixture contains mathematics and is not known to be free of Hendrycks MATH overlaps. The dev set is for checkpoint selection, not an untouched test-set performance claim. If strong contamination claims matter, audit the upstream SFT training mixture against the Hendrycks problem text.

## Preflight

Kaggle: T4 x2, Internet enabled. Colab: T4 or better, Internet enabled. Confirm hardware and CUDA driver first:

```bash
nvidia-smi -L
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
```

The existing project's `uv.lock` targets CUDA 13 and its `doctor` requires driver R580+. If the driver is older, do **not** force this lock onto the runtime; first establish a compatible pinned environment. Following setup, run:

```bash
uv run --locked --no-sync posttrain-math doctor
```

## Download both full-model sources and prepare MATH

```bash
uv run --locked --no-sync posttrain-math model download \
  --repo-id allenai/OLMo-2-0425-1B-SFT \
  --revision 0d85a3d037876ce6ac7d4311d994400fc66ac27f \
  --output-dir models/olmo-2-0425-1b-sft

uv run --locked --no-sync posttrain-math data download
uv run --locked --no-sync posttrain-math data prepare --seed 42 --dev-ratio 0.1
```

To compare with the original base or an original-base LoRA checkpoint, also run `posttrain-math model download` with no model arguments. The latter restores `models/olmo-2-0425-1b` at its pinned original-base revision. To evaluate a previous LoRA checkpoint, extract its adapter directory from the archived SFT run and pass that directory to `--model`; its base must also be available locally.

## A. Evaluate the unmodified official SFT before doing any extra training

```bash
uv run --locked --no-sync posttrain-math eval \
  --model models/olmo-2-0425-1b-sft \
  --data-dir data/processed \
  --split dev --levels 1 2 3 \
  --prompt boxed-1shot \
  --max-new-tokens 512 --batch-size 1 \
  --output-dir runs/eval-official-sft-l123-1shot
```

Report: `num_correct/266`, `accuracy`, `boxed_output_rate`, `numeric_output_rate`, and `accuracy_by_level`. Save **both** `metrics.json` and `predictions.jsonl`. Run the same command with prior `checkpoint-302` and `checkpoint-453`, changing model/output paths only, to obtain a paired comparison under the *same* prompt (their historical `boxed` results must not be compared as though the prompt were equal).

A small `--limit 4` smoke test is permissible before full evaluation but do not confuse it with the final 266-question result.

## B. Optional: LoRA SFT of the *official full SFT* on the same L1–3 numeric cohort

This is **additional** SFT on top of official Tülu SFT, not from the original pretrained base. Train only if Stage A indicates a need; do not require it for the initial comparison.

Single GPU (Colab):

```bash
uv run --locked --no-sync posttrain-math train sft \
  --model models/olmo-2-0425-1b-sft \
  --data-dir data/processed --output-dir runs/official-sft-plus-l123 \
  --prompt boxed-1shot --levels 1 2 3 \
  --epochs 3 --learning-rate 2e-4 --scheduler cosine --warmup-ratio 0.03 \
  --peft lora --lora-r 16 --lora-alpha 32 --lora-dropout 0.05 \
  --lora-target-modules all-linear --optim adamw_torch \
  --weight-decay 0 --batch-size 2 --global-batch-size 16 \
  --precision fp16 --max-length 2048 --overlong-policy drop \
  --gradient-checkpointing --max-grad-norm 1 \
  --save-strategy epoch --save-total-limit 4 --seed 42
```

Two GPUs (Kaggle T4 x2): use **the same options** but prefix the command with `uv run --locked --no-sync python -m torch.distributed.run --standalone --nproc-per-node=2 -m posttrain_math train sft ...` (rather than `posttrain-math train sft ...`). With per-device batch 2 and global batch 16, the trainer chooses gradient accumulation 4 for two GPUs versus 8 for one GPU.

Evaluate each epoch with `posttrain-math eval --model runs/official-sft-plus-l123/checkpoint-<step> --split dev --levels 1 2 3 --prompt boxed-1shot --max-new-tokens 512`. The adapter resolver will locate `models/olmo-2-0425-1b-sft` in a different machine/runtime if its `source.json` matches the pinned revision.

## Fair-comparison and RL requirements

1. Same MATH prepare contract (`seed=42`, `dev_ratio=0.1`), numeric-eligible L1–3 dev, prompt, `max_new_tokens=512`, greedy decoding and `boxed-numeric-v1` verifier for all rows.
2. Separate the conditions clearly: **official SFT as-is**; **official SFT + L1–3 LoRA**; **original-base + L1–3 LoRA**. Changes in initialization and supervision history are **not** controlled by matching inference prompts alone.
3. Prefer a **train-side**, held-out-from-dev sampled `G=16` probe after greedy evaluation before choosing an RL start. Greedy format/accuracy by themselves do not measure signal-bearing groups.
4. The current `train_rl` expects an **existing LoRA adapter**, not a full-model SFT checkpoint. Stage B produces one. If you want to run RL *directly* on the official SFT with no extra SFT, bootstrap a zero-update LoRA adapter in a separate, tested step first; do not silently treat a full checkpoint as an adapter.
5. Preserve the output `metrics.json`, `predictions.jsonl`, run configuration, source manifests and exact Git SHA. Do not touch the test split for model selection.
