# posttrain-math

Reproducible post-training experiments for mathematical language models. The
reference pipeline uses OLMo 2 1B and Hendrycks MATH for deterministic data
preparation, completion-only SFT, boxed-numeric evaluation, and RLVR.

The central evaluation/reward contract is deliberately narrow: **boxed-numeric-v1**.
Gold solutions are eligible only when they contain exactly one well-formed
`\boxed{...}` whose content matches the numeric grammar below. Model outputs are scored
by the final `\boxed{...}` marker with a small deterministic parser based on
Python `decimal.Decimal` and `fractions.Fraction`. No symbolic or natural-language verifier participates
in the primary RL reward.


The research focus is not to assume that outcome RLVR directly supervises
reasoning. The project separates two questions: **when do policy-gradient
estimators use sparse verifier rewards efficiently, and when do higher terminal
rewards correspond to textual intermediate states from which future success is
actually more likely?** Training remains outcome-only; reasoning-path analysis is
a separate diagnostic. See [`docs/rl_research.md`](docs/rl_research.md).

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

## boxed-numeric-v1 cohort

`data prepare` retains the full rows used by SFT and adds deterministic cohort
metadata:

- `gt_boxed`
- `gt_numerator`
- `gt_denominator`
- `numeric_eligible`
- `numeric_exclusion`

A gold row is eligible when its solution contains exactly one literal `\boxed`
marker, that marker is well-formed and non-empty, and its content successfully
parses under this grammar after stripping only leading/trailing whitespace:

```text
digits   := [0-9]+
number   := ["-"] digits ["." digits] ["\%"]
fraction := ["-"] "\frac{" digits "}{" digits "}"
answer   := number | fraction
```

The stripped content must full-match `answer`; no internal whitespace is removed.
Numbers are parsed exactly with `decimal.Decimal`, divided by 100 when the
optional escaped `\%` suffix is present, then converted to `fractions.Fraction`.
LaTeX fractions are constructed directly with `Fraction(numerator, denominator)`;
its native `ZeroDivisionError` is caught as an invalid parse. Both paths return
canonical `Fraction` values. Examples include `42`, `-17`, `0.5`, `50\%`,
`\frac{3}{5}`, and `-\frac{3}{5}`. A plus sign, a sign inside fraction braces,
slash fractions (`3/5`), bare `%`, scientific notation, radicals, symbols, units,
tuples, intervals, equations, and prose are outside this grammar.

For the pinned 12,500-row source, the cohort audit is:

| split | source rows | boxed-numeric-v1 | retention |
| --- | ---: | ---: | ---: |
| source train | 7,500 | 5,586 | 74.48% |
| test | 5,000 | 3,686 | 73.72% |
| combined | 12,500 | 9,272 | 74.176% |

These counts were recomputed with `data prepare` using the pinned raw files and
the current parser. `data download` verifies their revision and SHA256 before
reuse. The generated `data/processed/manifest.json` records `source_train`,
processed `train`/`dev`, `test`, and `combined` audits; combined counts each source
row once. No cohort row count is hardcoded in data preparation.

| exclusion | source train | test | combined |
| --- | ---: | ---: | ---: |
| `non_numeric_boxed` | 1,796 | 1,224 | 3,020 |
| `multiple_boxed` | 114 | 90 | 204 |
| `empty_boxed` | 2 | 0 | 2 |
| `malformed_boxed` | 2 | 0 | 2 |
| `no_boxed` | 0 | 0 | 0 |
| `zero_denominator` | 0 | 0 | 0 |
| total excluded | 1,914 | 1,314 | 3,228 |

The processed manifest records exclusion counts and `type × level` source/cohort
shares so selection drift is explicit. The regular train/dev split is still
stratified on `type × level`; SFT can therefore continue to use the complete
training split, while evaluation and RL select `numeric_eligible == True`.

## Verifier contract

For a model completion, only the **final** `\boxed` marker is considered. If the
final marker is malformed or its content fails numeric parsing, reward is `0`.
The verifier never falls back to an earlier box or a number in prose. It compares
only the canonical prediction and gold `Fraction` values: equality gives `1`,
otherwise `0`. There is no tolerance or symbolic equivalence.

Examples:

```text
gold canonical Fraction = 1/2
prediction = \boxed{\frac{2}{4}} -> correct
prediction = \boxed{0.5}       -> correct
prediction = \boxed{50\%}      -> correct
prediction = \boxed{2/4}       -> incorrect
prediction = \boxed{\sqrt{1/4}} -> incorrect
prediction = "answer is 1/2"   -> incorrect
prediction = \boxed{0.5} then \boxed{1/0} -> incorrect (no fallback)
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

Evaluation automatically selects the fixed `boxed-numeric-v1` cohort and writes
`predictions.jsonl` plus `metrics.json`. Metrics include accuracy, boxed-output
rate, `numeric_output_rate`, and accuracy by level/type. Prediction records use
`prediction_numeric`, and metrics count valid numeric outputs in `num_pred_numeric`.
Re-run `data prepare` to regenerate processed metadata before evaluation or RL;
older processed datasets and evaluation metrics are not compatible with this contract.

## RLVR

Full model training uses maintained TRL trainers; this repository keeps small
reference implementations of estimator/objective formulas so the semantics are
unit-testable without maintaining a second RL framework.

The controlled first-stage matrix is:

| Method | Backend | Status / contract |
| --- | --- | --- |
| REINFORCE | project reference objective | mathematical baseline only; no custom 1B trainer |
| RLOO | TRL `RLOOTrainer` | leave-one-out reward baseline, no advantage normalization |
| GRPO | TRL `GRPOTrainer` | original `loss_type="grpo"`, `scale_rewards="group"` |
| Dr.GRPO | TRL `GRPOTrainer` | `loss_type="dr_grpo"`, `scale_rewards="none"` |
| PPO | reference clipped objective | production backend intentionally deferred |

RLOO, GRPO, and Dr.GRPO all use `num_iterations=1`. Controlled estimator
comparisons default to `beta=0`, `temperature=1.0`, `top_p=1.0`, `top_k=0`, and
disabled dropout. This keeps the first experiments on-policy and avoids silently
mixing estimator changes with nucleus/temperature truncation or KL penalties.
Every run records the explicit estimator contract and pinned TRL version in
`run_config.json`.

Run a backend with the common launcher:

```bash
# RLOO
bash scripts/launch_rl.sh rloo auto \
  --model runs/olmo2-1b-lora-sft-v1/final-model \
  --output-dir runs/olmo2-1b-rloo-v1 \
  --max-steps 100

# original GRPO
bash scripts/launch_rl.sh grpo auto \
  --model runs/olmo2-1b-lora-sft-v1/final-model \
  --output-dir runs/olmo2-1b-grpo-v1 \
  --max-steps 100

# Dr.GRPO
bash scripts/launch_rl.sh dr-grpo auto \
  --model runs/olmo2-1b-lora-sft-v1/final-model \
  --output-dir runs/olmo2-1b-dr-grpo-v1 \
  --max-steps 100
```

`scripts/launch_grpo.sh` remains as a backwards-compatible wrapper around
`launch_rl.sh grpo`.

Fair comparisons should be reported against generated completion/token budget,
verifier calls, and GPU time in addition to optimizer steps. For binary reward
and group size `G`, the predicted fraction of groups with both successes and
failures is `1 - p^G - (1-p)^G`; comparing that prediction with observed useful
groups is part of the planned reward-topology analysis.

### Continuation-value reasoning probe

Training reward remains **terminal-only**. A separate frozen-policy probe
estimates the empirical continuation value of a textual prefix by sampling fresh
continuations and applying the same `boxed-numeric-v1` terminal verifier.

First estimate SFT prompt success probabilities on dev and freeze a frontier
prompt set:

```bash
uv run --locked --no-sync python -m posttrain_math.reasoning_probe prompt-success \
  --model runs/olmo2-1b-lora-sft-v1/final-model \
  --split dev \
  --rollouts 8 \
  --limit-prompts 128 \
  --output-dir runs/probes/sft-dev-prompt-success
```

Then evaluate fixed 0/25/50/75% prefixes of the pre-box span on SFT and an RL checkpoint using the
same selection artifact:

```bash
uv run --locked --no-sync python -m posttrain_math.reasoning_probe prefix-value \
  --model runs/olmo2-1b-grpo-v1/final-model \
  --split dev \
  --selection runs/probes/sft-dev-prompt-success/prompt_success.parquet \
  --min-p 0.125 --max-p 0.875 \
  --limit-prompts 32 \
  --branch-rollouts 8 \
  --output-dir runs/probes/grpo-dev-prefix-value
```

This probe measures **continuation success under the policy**, not hidden-state
interpretability, causal faithfulness of chain-of-thought, or deterministic
step correctness. Keyword-defined "aha moments" and process judges are deferred
until this fixed-prefix pilot demonstrates a signal worth explaining. The full
research contract is in [`docs/rl_research.md`](docs/rl_research.md).

Evaluate a sequence of RL checkpoints with:

```bash
bash scripts/eval_checkpoints.sh auto runs/olmo2-1b-grpo-v1
```
## Artifact contract

Local generated artifacts belong under `runs/<experiment>/`. Successful SFT and
RL runs create `provenance.json` containing the git commit/dirty state,
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
suite on pushes to `master` and on pull requests.
