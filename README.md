# postrain-math-lab

MATH post-training lab with local-only model execution, deterministic data preparation, LoRA SFT, evaluation, and a cloud stack prepared for quantization and vLLM-backed RL/evaluation work.

## Cloud bootstrap (Kaggle / Colab)

Enable an NVIDIA GPU runtime, clone the repo, then run one command:

```bash
bash scripts/bootstrap_cloud.sh
```

The script verifies/installs `uv`, creates an isolated managed-Python-3.12 `.venv`, installs the repo with the GPU-compatible PyTorch backend plus vLLM/bitsandbytes/dev dependencies, downloads the model and MATH dataset, prepares the deterministic split, runs tests/environment checks, and executes a two-example GPU evaluation smoke test.

The bootstrap is safe to rerun: a valid existing `.venv`, complete model download, and complete raw dataset are reused.

## Local-resource contract

The bootstrap downloads:

```text
Qwen/Qwen3-1.7B-Base -> models/qwen3-1.7b-base/
EleutherAI/hendrycks_math -> data/raw/
                           -> data/processed/
```

Training and evaluation accept local model/data paths only. `models/`, `data/`, and `.venv/` are intentionally ignored by Git.

Manual download commands, when needed:

```bash
.venv/bin/postrain-math model download
.venv/bin/postrain-math data download
.venv/bin/postrain-math data prepare
```

## Sanity training

LoRA one-batch overfit:

```bash
.venv/bin/postrain-math train overfit-one-batch \
  --peft lora \
  --precision auto
```

Defaults: LoRA `r=16`, `alpha=32`, `dropout=0.05`, `target_modules=all-linear`, learning rate `2e-4`.

Full-parameter sanity path:

```bash
.venv/bin/postrain-math train overfit-one-batch \
  --peft none \
  --learning-rate 1e-4 \
  --precision auto
```

## LoRA SFT

```bash
.venv/bin/postrain-math train sft \
  --peft lora \
  --precision auto \
  --gradient-checkpointing
```

Default SFT learning rate is `2e-4` for LoRA and `2e-5` for full-parameter SFT.

## Evaluation

```bash
.venv/bin/postrain-math eval \
  --split dev \
  --prompt boxed
```

A saved LoRA adapter can be evaluated by passing its local `final-model` directory; its adapter metadata points to the local base-model directory used during training.
