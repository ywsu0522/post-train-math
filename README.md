# posttrain-math

Reproducible post-training experiments for mathematical language models. The current pipeline uses a local OLMo 2 1B base model and the Hendrycks MATH dataset to provide deterministic data preparation, completion-only LoRA SFT, and boxed-answer evaluation.

## Scope

The repository currently owns:

- reproducible dependency metadata and a committed `uv.lock`;
- explicit model and dataset download commands;
- deterministic train/dev/test preparation;
- a one-batch overfit sanity check and LoRA/full-parameter SFT;
- single-/multi-GPU Hugging Face evaluation with mathematical answer verification;
- GRPO post-training from an SFT LoRA adapter using math-verify rewards.

GRPO v1 uses Transformers generation in the locked training environment. Quantization and vLLM rollouts remain future work; vLLM stays in a separate environment so its CUDA/PyTorch wheel constraints cannot change the locked training environment.

## Naming and directories

| Name | Purpose | Git policy |
| --- | --- | --- |
| `posttrain-math` | repository, Python distribution, and CLI | committed |
| `posttrain_math` | importable Python package | committed |
| `data/` | downloaded raw data and prepared Parquet splits | ignored |
| `models/` | downloaded base models | ignored |
| `runs/` | evaluations, training logs, checkpoints, and final adapters | ignored |
| `examples/reference_runs/` | small historical outputs kept as examples | committed |

Run commands from the repository root. Relative paths in the CLI follow this directory contract.

## Dependency contract

`pyproject.toml` declares constraints; `uv.lock` records the exact cross-platform resolution and must be committed. `.venv/` is always local and must not be committed.

The project requires Python 3.12 and `uv >= 0.12.6`.

On Windows, after changing `pyproject.toml`:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/update_lock.ps1
git add pyproject.toml uv.lock
```

On Linux or macOS:

```bash
bash scripts/update_lock.sh
git add pyproject.toml uv.lock
```

For ordinary local checks, use the committed lock without resolving new versions:

```powershell
uv sync --locked --group dev
uv run --locked pytest -q
uv run --locked ruff check src tests
```

## Colab T4 workflow

Enable a T4 GPU runtime, clone the GitHub repository, change into it, and run:

```bash
bash scripts/bootstrap_colab.sh
```

Bootstrap performs these steps once per fresh Colab runtime:

1. install or update `uv`;
2. verify the committed lock and create `.venv/`;
3. verify CUDA is visible inside the locked environment;
4. download the model and raw dataset;
5. prepare deterministic data splits;
6. run unit tests, an environment check, and a two-example GPU evaluation.

The downloads are resumable/reusable when their manifests and expected files are present. Training and evaluation never download resources implicitly.

The resulting local resources are:

```text
allenai/OLMo-2-0425-1B -> models/olmo-2-0425-1b/
EleutherAI/hendrycks_math -> data/raw/
                            -> data/processed/
```


Qwen remains available through the generic model download interface when an explicit comparison is wanted:

```bash
uv run --locked posttrain-math model download \
  --repo-id Qwen/Qwen3-1.7B-Base \
  --output-dir models/qwen3-1.7b-base
```

The same resource steps can be run manually:

```bash
uv run --locked posttrain-math model download
uv run --locked posttrain-math data download
uv run --locked posttrain-math data prepare
uv run --locked posttrain-math environment
```

## Single- and dual-GPU runtime

The Python core is shared across Colab and Kaggle. `scripts/launch_gpu.sh`
selects one process per GPU:

```bash
# Colab or a forced single-GPU run
bash scripts/launch_gpu.sh 1 eval --split dev --prompt boxed --batch-size 2

# Kaggle T4x2
bash scripts/launch_gpu.sh 2 eval --split dev --prompt boxed --batch-size 2

# Detect all visible GPUs
bash scripts/launch_gpu.sh auto eval --split dev --prompt boxed --batch-size 2
```

Multi-GPU evaluation replicates the model once per GPU, shards the fixed
evaluation cohort by row index, writes per-rank JSONL shards, and merges them
on rank 0 into the same `predictions.jsonl` and `metrics.json` contract used by
single-GPU evaluation.

SFT uses Hugging Face Trainer/Accelerate DDP when launched with more than one
process. The default target global batch is 16, so with per-device batch 2 the
resolved gradient accumulation is 8 on one GPU and 4 on two GPUs.

## Training

Start with the one-batch LoRA overfit check:

```bash
uv run --locked posttrain-math train overfit-one-batch \
  --peft lora \
  --precision auto
```

Then run LoRA SFT:

```bash
uv run --locked posttrain-math train sft \
  --output-dir runs/olmo2-1b-lora-sft-v1 \
  --peft lora \
  --precision auto \
  --gradient-checkpointing
```

Defaults are LoRA `r=16`, `alpha=32`, `dropout=0.05`, `target_modules=all-linear`, and learning rate `2e-4`. Full-parameter SFT defaults to `2e-5`.

Each SFT run stores Hugging Face `checkpoint-*` directories and a `final-model/` under its output directory. Resume an interrupted run explicitly:

```bash
uv run --locked posttrain-math train sft \
  --output-dir runs/my-experiment \
  --resume-from-checkpoint runs/my-experiment/checkpoint-100
```

## Evaluation

Evaluation uses the fixed parseable-gold cohort materialized by `data prepare` and reports overall accuracy plus breakdowns by MATH level and problem type. Use the same prompt strategy and generation budget for every model snapshot.

Evaluate the base model on the dev split:

```bash
uv run --locked posttrain-math eval \
  --split dev \
  --prompt boxed
```

Evaluate a saved checkpoint or final LoRA adapter with `--model`:

```bash
uv run --locked posttrain-math eval \
  --model runs/my-experiment/checkpoint-100 \
  --split dev \
  --prompt boxed
```

Evaluate the final adapter by pointing `--model` at its `final-model/` directory:

```bash
uv run --locked posttrain-math eval \
  --model runs/my-experiment/final-model \
  --split dev \
  --prompt boxed
```

## GRPO post-training v1

The first RL stage uses Group Relative Policy Optimization (GRPO) starting from
an SFT LoRA adapter. The rollout reward uses the same locked `math-verify`
contract as evaluation: a correct boxed answer receives `1.0`; an incorrect but
parseable boxed answer receives a small configurable format reward (default
`0.05`); all other outputs receive `0.0`.

GRPO uses sampled rollouts (`num_generations=4` by default) while the benchmark
evaluator remains deterministic greedy decoding. v1 uses `beta=0.0`, so no
reference-model copy is required.

Smoke test:

```bash
bash scripts/launch_grpo.sh 1 \
  --model runs/olmo2-1b-lora-sft-v1/final-model \
  --output-dir runs/olmo2-1b-grpo-smoke \
  --limit-prompts 32 \
  --max-steps 5
```

Kaggle T4x2 run:

```bash
bash scripts/launch_grpo.sh 2 \
  --model runs/olmo2-1b-lora-sft-v1/final-model \
  --output-dir runs/olmo2-1b-grpo-v1 \
  --max-steps 100
```

Evaluate GRPO checkpoints with the same deterministic evaluator:

```bash
bash scripts/eval_checkpoints.sh 2 runs/olmo2-1b-grpo-v1
```

## Keeping Colab checkpoints

The default `runs/` directory is fast Colab-local storage and disappears when the runtime is recycled. At the end of a run, copy the whole experiment directory to Google Drive or download it to Windows; keeping the entire directory preserves configuration, logs, intermediate checkpoints, tokenizer files, and the final adapter together.

For crash resilience, mount Google Drive and pass a Drive path directly, accepting slower checkpoint writes:

```bash
uv run --locked posttrain-math train sft \
  --output-dir /content/drive/MyDrive/posttrain-math/runs/my-experiment \
  --peft lora \
  --precision auto \
  --gradient-checkpointing
```

Do not copy downloaded `models/` or `data/` unless avoiding their next-runtime download is worth the storage; both can be recreated from their recorded source revisions.

## Optional vLLM environment

When rollout or server-backed evaluation work begins:

```bash
bash scripts/setup_vllm.sh
```

This creates `.venv-vllm/` and installs the pinned vLLM version using the GPU-compatible PyTorch backend. It is separate from `.venv/` by design.
