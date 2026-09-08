# posttrain-math

Reproducible post-training experiments for mathematical language models. The
reference pipeline uses OLMo 2 1B and Hendrycks MATH for deterministic data
preparation, completion-only SFT, exact-rational evaluation, and RLVR.

The central evaluation/reward contract is deliberately narrow: **exact-rational-v1**.
Gold solutions are eligible only when they contain exactly one well-formed
`\boxed{...}` whose content is an exact rational number. Model outputs are scored
by the final `\boxed{...}` marker with a small deterministic parser based on
Python `fractions.Fraction`. No symbolic or natural-language verifier participates
in the primary RL reward.

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
fresh clone. The legacy `posttrain-math environment` command remains as an alias.

## Pinned external resources

Reference experiments do not follow mutable Hugging Face `main` branches.

- model: `allenai/OLMo-2-0425-1B` at pinned revision
  `13cece9360d59bb9db636273ea8d000b67fcc27b`;
- dataset: the consolidated Hendrycks MATH mirror
  `DigitalLearningGmbH/MATH-lighteval` at pinned revision
  `f06834690385b29df31ccc717250746a3ba0322b`.

The pinned dataset revision contains exactly two source Parquet files:
`data/train-00000-of-00001.parquet` (7,500 rows) and
`data/test-00000-of-00001.parquet` (5,000 rows). Each row already contains
`problem`, `solution`, `type`, and `level`; the project does not download or
reassemble seven per-subject configs.

Download and prepare once:

```bash
uv run --locked --no-sync posttrain-math model download
uv run --locked --no-sync posttrain-math data download
uv run --locked --no-sync posttrain-math data prepare
```

Each download records the upstream resolved commit and file SHA256 values.
`models/`, `data/`, and `runs/` remain gitignored.

## exact-rational-v1 cohort

`data prepare` retains the full rows used by SFT and adds deterministic cohort
metadata:

- `gt_boxed`
- `gt_numerator`
- `gt_denominator`
- `rational_eligible`
- `rational_exclusion`

A gold row is eligible when its solution contains exactly one well-formed,
non-empty `\boxed{...}` marker and the boxed content matches one of:

```text
42
-17
3/5
-3/5
\frac{3}{5}
-\frac{3}{5}
```

Signed numerators/denominators are accepted and canonicalized with
`fractions.Fraction`; denominator zero is rejected. Decimal, radical, symbolic,
unit-bearing, tuple, interval, percentage, equation, and prose answers are out of
domain by design.

For the pinned 12,500-row source, the cohort audit is:

| split | source rows | exact-rational-v1 | retention |
| --- | ---: | ---: | ---: |
| source train | 7,500 | 5,448 | 72.64% |
| test | 5,000 | 3,592 | 71.84% |
| combined | 12,500 | 9,040 | 72.32% |

The processed manifest records exclusion counts and `type × level` source/cohort
shares so selection drift is explicit. The regular train/dev split is still
stratified on `type × level`; SFT can therefore continue to use the complete
training split, while evaluation and RL select `rational_eligible == True`.

## Verifier contract

For a model completion, only the **final** `\boxed` marker is considered. If the
final marker is malformed or its content is outside the exact-rational grammar,
the prediction is invalid. The verifier never falls back to an earlier box, a
last number in prose, a decimal approximation, or symbolic equivalence.

Examples:

```text
gold = 1/2
prediction = \boxed{2/4}       -> correct
prediction = \boxed{0.5}       -> incorrect
prediction = \boxed{\sqrt{1/4}} -> incorrect
prediction = "answer is 1/2"   -> incorrect
```

The same parser is used for base OLMo, SFT checkpoints, RL checkpoints, offline
evaluation, and RL rewards. Primary rewards are binary `{0, 1}`.

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
summaries, plots, and `provenance.json` under the selected run directory.

## Evaluation

```bash
bash scripts/launch_gpu.sh auto eval \
  --model runs/olmo2-1b-lora-sft-v1/final-model \
  --split dev \
  --prompt boxed \
  --output-dir runs/evals/olmo2-1b-lora-sft-v1
```

Evaluation automatically selects the fixed `exact-rational-v1` cohort and writes
`predictions.jsonl` plus `metrics.json`. Metrics include accuracy, boxed-output
rate, exact-rational-output rate, and accuracy by level/type.

## RLVR

The intended method progression is deliberately incremental:

```text
REINFORCE -> RLOO -> original GRPO -> Dr.GRPO -> selected DAPO components
```

The current implemented group-relative baseline is **original GRPO**. The TRL
configuration explicitly sets `loss_type="grpo"`, `num_iterations=1`, and
`scale_rewards="group"`; this avoids silently inheriting TRL's DAPO-style loss
default. Reward is the same exact-rational binary verifier used by evaluation.

```bash
bash scripts/launch_grpo.sh auto \
  --model runs/olmo2-1b-lora-sft-v1/final-model \
  --output-dir runs/olmo2-1b-grpo-v1 \
  --max-steps 100
```

`run_config.json` records the rollout budget, number of generations, prompt
groups per update, verifier name, and explicit GRPO loss settings. REINFORCE and
RLOO should be added as separate baseline implementations before introducing
Dr.GRPO/DAPO mechanisms so changes in optimization can be attributed cleanly.

Evaluate a sequence of checkpoints with:

```bash
bash scripts/eval_checkpoints.sh auto runs/olmo2-1b-grpo-v1
```

## Artifact contract

Local generated artifacts belong under `runs/<experiment>/`. Successful SFT and
GRPO runs create `provenance.json` containing the git commit/dirty state,
`uv.lock` hash, model and dataset source manifests, processed-data manifest, and
hardware/runtime report. Keep configuration, logs, metrics, checkpoints needed
for resume, and the final model/adapter together.

A local workstation keeps `runs/` on its normal filesystem. Hosted notebook
filesystems are ephemeral, so persistence is handled by the notebook workflow:

- [Colab workflow](docs/colab.md)
- [Kaggle workflow](docs/kaggle.md)
- [Artifact and cache policy](docs/artifacts.md)

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

Dependency maintenance uses `uv lock`. Installation uses `uv sync --locked`.
Normal execution uses `uv run --locked --no-sync`, which keeps experiment
execution from mutating the environment.

GitHub Actions runs `uv lock --check`, Ruff, and the CPU-compatible unit test
suite on pushes and pull requests. GPU execution remains a separate hardware
validation step because hosted CI runners do not provide the reference NVIDIA
runtime.
