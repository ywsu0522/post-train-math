# posttrain-math

Reproducible post-training experiments for mathematical language models. The
current pipeline uses OLMo 2 1B and Hendrycks MATH for deterministic data
preparation, completion-only SFT, GRPO, and boxed-answer evaluation.

## Reference environment

The reference GPU runtime is Linux x86_64 with:

- Python 3.12;
- the committed `uv.lock`;
- PyTorch 2.13.0, whose locked Linux dependency set uses CUDA 13.x;
- NVIDIA driver R580 or newer;
- NVIDIA compute capability 7.5 or newer (Turing / SM75+).

A T4 (SM75) is supported and uses FP16. Newer GPUs use BF16 automatically when
PyTorch reports native BF16 support. Multi-GPU runs use one process per visible
GPU and NCCL through `torchrun`.

The CUDA 13 architecture and driver requirements are documented in
[`docs/environment.md`](docs/environment.md).

## Install

Installation is an explicit user action. Runtime scripts never run `uv sync`.

```bash
git clone https://github.com/ywsu0522/post-train-math.git
cd post-train-math

uv sync --locked --group dev
uv run --locked --no-sync pytest -q
uv run --locked --no-sync posttrain-math doctor
```

`doctor` checks the installed Torch CUDA build, NVIDIA driver, visible GPUs,
compute capability, VRAM, BF16 support, and the presence of local model/data
resources. Model/data absence is informational so the command is useful on a
fresh clone.

The legacy `posttrain-math environment` command remains as an alias.

## Pinned external resources

Reference experiments do not follow mutable Hugging Face `main` branches.

- model: `allenai/OLMo-2-0425-1B` at pinned revision `13cece9360d59bb9db636273ea8d000b67fcc27b`;
- dataset: `EleutherAI/hendrycks_math` at
  `21a5633873b6a120296cce3e2df9d5550074f4a3`.

Download and prepare once:

```bash
uv run --locked --no-sync posttrain-math model download
uv run --locked --no-sync posttrain-math data download
uv run --locked --no-sync posttrain-math data prepare
```

Each download records the upstream resolved commit in a local manifest.
`models/`, `data/`, and `runs/` remain gitignored.

Pinning protects experiments from upstream drift. It cannot make an uncached
remote service available during an outage. For resilient hosted runs, keep
`HF_HOME` on persistent storage or prefetch before training. See
[`docs/artifacts.md`](docs/artifacts.md).

## Single- and multi-GPU execution

The Python core is platform-independent. The thin launch scripts only select
one process per GPU; they do not install dependencies.

```bash
# one GPU
bash scripts/launch_gpu.sh 1 eval --split dev --prompt boxed --batch-size 2

# two GPUs
bash scripts/launch_gpu.sh 2 eval --split dev --prompt boxed --batch-size 2

# all visible GPUs
bash scripts/launch_gpu.sh auto eval --split dev --prompt boxed --batch-size 2
```

SFT uses Hugging Face Trainer/Accelerate DDP when launched with multiple
processes. The target global batch is preserved through gradient accumulation.

## SFT

First validate the completion-only data contract:

```bash
uv run --locked --no-sync posttrain-math train inspect
```

Then run a one-batch LoRA overfit sanity check:

```bash
uv run --locked --no-sync posttrain-math train overfit-one-batch \
  --peft lora \
  --precision auto
```

Run SFT:

```bash
bash scripts/launch_gpu.sh auto train sft \
  --output-dir runs/olmo2-1b-lora-sft-v1 \
  --peft lora \
  --precision auto \
  --gradient-checkpointing
```

SFT writes Trainer checkpoints plus `final-model/`, configuration, logs,
summaries, and plots under the selected run directory.

## Evaluation

```bash
bash scripts/launch_gpu.sh auto eval \
  --model runs/olmo2-1b-lora-sft-v1/final-model \
  --split dev \
  --prompt boxed \
  --output-dir runs/evals/olmo2-1b-lora-sft-v1
```

Evaluation writes `predictions.jsonl` and `metrics.json`. The current benchmark
metric requires a parseable final `\boxed{...}` answer and verifies its
mathematical equivalence with the locked `math-verify` version.

## GRPO

GRPO remains a separate module and uses the same answer-verification contract.

```bash
bash scripts/launch_grpo.sh auto \
  --model runs/olmo2-1b-lora-sft-v1/final-model \
  --output-dir runs/olmo2-1b-grpo-v1 \
  --max-steps 100
```

Evaluate a sequence of checkpoints with:

```bash
bash scripts/eval_checkpoints.sh auto runs/olmo2-1b-grpo-v1
```

## Artifact contract

Local generated artifacts belong under `runs/<experiment>/`. Keep source
manifests, configuration, logs, metrics, checkpoints needed for resume, and the
final model/adapter together.

A local workstation keeps `runs/` on its normal filesystem. Hosted notebook
filesystems are ephemeral, so persistence is handled by the notebook workflow,
without Colab/Kaggle branches in the Python package:

- [Colab workflow](docs/colab.md)
- [Kaggle workflow](docs/kaggle.md)
- [Artifact and cache policy](docs/artifacts.md)

For public reproducibility, publish the final adapter/model and the small
configuration/metric manifests to a versioned model repository (for example,
Hugging Face Hub). Preserve large intermediate checkpoints only when they are
needed for resume or for a specific analysis.

## Repository directories

| Path | Purpose | Git policy |
| --- | --- | --- |
| `src/posttrain_math/` | reusable Python implementation | committed |
| `tests/` | CPU-testable contracts | committed |
| `scripts/` | thin generic launch/orchestration helpers | committed |
| `docs/` | environment and hosted-notebook procedures | committed |
| `data/` | downloaded and derived dataset files | ignored |
| `models/` | downloaded base-model snapshot | ignored |
| `runs/` | training/evaluation artifacts | ignored |
| `examples/reference_runs/` | small historical reference outputs | committed |

Dependency maintenance uses `uv lock`. Installation uses
`uv sync --locked`. Normal execution uses `uv run --locked --no-sync`, which
keeps experiment execution from mutating the environment.
