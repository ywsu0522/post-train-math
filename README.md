# post-train-math

Pinned Hendrycks MATH data, the frozen `boxed-numeric` cohort, and RLOO
training with TRL + PEFT for two pinned checkpoints. Python 3.12 and uv are
required. Data preparation and tests run on CPU; evaluation and training require CUDA.

## Frozen data contract

The order is fixed in code and `cohort/manifest.json`:

1. Verify the original source files and their SHA256 hashes.
2. Preserve **all 7,500 original train problems** and **all 5,000 original test
   problems** in their original splits and order. Reject overlapping problem texts.
3. Apply the **same** `boxed-numeric` rule independently within train and test.
   Use every eligible original training problem for RL.

The source is `DigitalLearningGmbH/MATH-lighteval` at
`f06834690385b29df31ccc717250746a3ba0322b`.

| Source file | SHA256 |
| --- | --- |
| `data/train-00000-of-00001.parquet` | `eca6e667f4305dd5e5ba09b4fd55e7f3174a0fbe361cdfd4c44758b593a76933` |
| `data/test-00000-of-00001.parquet` | `7dca8d6e41af88ecf82f2b5f36eb5530e083aaaa86ee325f62bd5c31535178c6` |

A gold solution is eligible only if it contains **exactly one literal `\boxed`
marker**, with a complete, balanced, nonempty braced argument. Additional
markers, including empty, nested, or malformed ones, exclude the whole solution.
After stripping outer whitespace, the entire boxed content must match:

```text
digits         := [0-9]+
number         := ["-"] digits ["." digits] ["\%"]
fraction_latex := ["-"] "\frac{" digits "}{" digits "}"
fraction_slash := ["-"] digits "/" digits
answer         := number | fraction_latex | fraction_slash
```

Internal whitespace, plus signs, signed denominators, zero denominators,
scientific notation, expressions, units, and symbolic answers are invalid.
Decimals are parsed exactly; percentages divide by 100. Accepted values are
canonicalized as `Fraction`.

| Split | Before filtering | Eligible |
| --- | ---: | ---: |
| train | 7,500 | 5,589 |
| test | 5,000 | 3,686 |

Processed Parquet files retain every original row plus eligibility and exact
gold metadata. Consumers select `numeric_eligible` within the requested split.
Preparation records provenance, counts, exclusions, and file hashes. Evaluation
and training reject files that differ from the frozen hashes, even if their
local manifest was edited to match. This preserves the frozen problem groups.

Prediction reward uses only the final boxed marker: 1 for exact rational equality,
0 otherwise. A malformed final box cannot fall back to an earlier box.

## Workflows

```bash
uv sync --locked --group dev

# 1. Download and annotate the original train/test data.
uv run --locked post-train-math data download
uv run --locked post-train-math data prepare

# 2. Download the pinned model checkpoints.
uv run --locked post-train-math model download --model-key olmo2
uv run --locked post-train-math model download --model-key granite

# 3. Evaluate the fixed 512 test questions with greedy decoding.
uv run --locked post-train-math eval --model-key olmo2 --output-dir runs/olmo2-test
uv run --locked post-train-math eval --model-key granite --output-dir runs/granite-test

# 4. Train a LoRA adapter with RLOO.
uv run --locked post-train-math train --model-key olmo2 --output-dir runs/olmo2-rloo --prompts-per-update 8 --rollouts 4 --per-device-train-batch-size 1 --gradient-accumulation-steps 32 --max-steps 200

# Evaluate the trained adapter.
uv run --locked post-train-math eval --model-key olmo2 --adapter-dir runs/olmo2-rloo/adapter --output-dir runs/olmo2-rloo-test
```

Evaluation always uses the original test cohort. It defaults to `--test-set fixed512`;
use `--test-set full` for all 3,686 eligible test questions. Use `--limit N` only for
a recorded smoke evaluation of the first N rows in the chosen set. Smoke reports
cannot be used to build learning curves. Model/data directories may be overridden with `--model-dir` and
`--data-dir`. New run output directories must be empty; explicit checkpoint
resume may reuse the original run directory. The module entry point is
`python -m post_train_math.cli`.

| Key | Checkpoint | Commit |
| --- | --- | --- |
| `olmo2` | `allenai/OLMo-2-0425-1B-SFT` | `0d85a3d037876ce6ac7d4311d994400fc66ac27f` |
| `granite` | `ibm-granite/granite-3.3-2b-instruct` | `707f574c62054322f6b5b04b6d075f0a8f05e0f0` |

Both models use their native chat templates and explicit fixed system prompts.
Training reads only the train cohort. The evaluation workflow reads only test.
Training saves `adapter/`, `train_report.json`, and the subset audit below; evaluation saves
`predictions.jsonl` and `eval_report.json`.

## Fixed test sample and checkpoint comparisons

The concrete test sample is committed in `cohort/testing-subset/samples.jsonl`.
Its manifest records the parent dataset hash, sampling rule, per-stratum counts,
sample-file hash, and ordered problem-ID hash. This is a subset of the original
test cohort, shared by all models, methods, training seeds, and checkpoints.

The 512 questions are allocated proportionally across all 35 `(type, level)`
groups using largest remainders. Within each group, questions are selected by
seed-42 SHA256 ranking, without replacement. Evaluation reconstructs the same
selection and rejects any mismatch with the frozen ordered-ID hash:

```text
174d7a6241e33c11edf799b8dd5dbfa73b9e75e539516cdd41531532f04922d6
```

Each group contributes at least four questions. Sample accuracy is the unweighted
fraction correct among these 512 questions; it is not a weighted estimate of the
full-test accuracy. Each checkpoint evaluates exactly the same questions with the
same greedy decoding and 2,048-token completion budget. Evaluation reports include:

| Metric | Definition; denominator is all evaluated questions |
| --- | --- |
| `accuracy` | Last boxed answer parses and equals the gold rational value exactly |
| `boxed_output_rate` | Last boxed marker has a balanced, nonempty braced argument, regardless of correctness or numeric validity |
| `numeric_boxed_output_rate` | Last boxed marker contains a valid boxed-numeric value, regardless of correctness |

A malformed final box counts as failure even if an earlier box was valid. Multiple
boxes are recorded separately in per-question `boxed_format`; the metrics above
follow the existing last-box reward rule. Every prediction stores its problem ID,
test-cohort index, completion, gold value, correctness, and formatting diagnostics.

Training defaults to 200 optimizer steps and saves `checkpoint-50`,
`checkpoint-100`, `checkpoint-150`, and `checkpoint-200`. In this RLOO recipe,
`num_iterations=1` and the batch/accumulation contract ensure exactly one optimizer
update per newly generated rollout batch, so these are also rollout iterations.
Earlier checkpoints are retained. They contain LoRA adapters and must be loaded
on top of the matching pinned base model.

A saved checkpoint can be evaluated in a new session before or after the training
run ends; its own `run_definition.json` and `trainer_state.json` provide provenance
without requiring a final `train_report.json`:

```bash
uv run --locked post-train-math eval --model-key olmo2 --adapter-dir runs/olmo2-rloo/checkpoint-50 --batch-size 8 --series-name RLOO --output-dir runs/olmo2-rloo-step50-test512
```

After all four planned checkpoints exist, evaluate them and the step-0 base model:

```bash
uv run --locked post-train-math eval-series --model-key olmo2 --run-dir runs/olmo2-rloo --steps 50 100 150 200 --batch-size 8 --series-name RLOO --output-dir runs/olmo2-rloo-test512
```

The command saves a report and predictions under `step-000000`, `step-000050`,
`step-000100`, `step-000150`, and `step-000200`, then creates `plots/scores.csv`,
`plots/learning_curves.png`, `plots/learning_curves.svg`, and `plots/analysis.json`.
The two panels show accuracy and boxed-output rate against RL rollout iterations;
the CSV also includes numeric-boxed rate and optimizer steps. Analysis records
endpoint changes in percentage points. Batch size should fit the target GPU.
Use `--no-include-base` to omit step 0, or a different `--steps` list for saved checkpoints.

Saved reports can be plotted again on CPU in another session, without running
inference. Reports from different methods can be passed together if they use the
same test IDs and decoding settings and have distinct `--series-name` labels:

```bash
uv run --locked post-train-math plot --reports runs/olmo2-rloo-test512/step-000000/eval_report.json runs/olmo2-rloo-test512/step-000050/eval_report.json runs/olmo2-rloo-test512/step-000100/eval_report.json runs/olmo2-rloo-test512/step-000150/eval_report.json runs/olmo2-rloo-test512/step-000200/eval_report.json --output-dir runs/olmo2-rloo-replot
```

Plotting rejects mixed question sets, token budgets, metric definitions, duplicate
steps within one series, and smoke evaluations. These commands prepare the
experiment pipeline; committed test samples and CPU smoke tests are not measured
GPU policy results.

The research artifact workflow consists of four connected modules:

| Module | Purpose | Artifacts to retain from an experiment |
| --- | --- | --- |
| `testing.py` | Reconstruct and verify the shared test questions | Committed sample and sampling manifest |
| `evaluation.py` + `cli.py eval` | Score each answer and bind results to a saved checkpoint | `predictions.jsonl`, `eval_report.json` |
| `cli.py eval-series` | Evaluate step 0 and the requested saved checkpoints consistently | One prediction/report pair per step |
| `curves.py` + `cli.py plot` | Replot reports on CPU and export counts, rates, and endpoint changes | `scores.csv`, PNG, SVG, `analysis.json` |

Keep the per-question predictions even after plotting: aggregate reports cannot
recover which questions improved or regressed. `analysis.json` contains endpoint
differences and source-report paths; it does not perform significance tests.
PNG is for quick inspection and SVG is for publication editing. Generated run
artifacts belong under the ignored `runs/` directory, not in the source commit.
The committed cohort manifests and tokenizer audits are reproducibility inputs.
This workflow currently supports greedy evaluation and individual series; it
does not compute pass@K, confidence intervals, or aggregate training seeds.
Use a distinct series name for each training run when plotting several runs.

RLOO uses TRL 1.12.0 and PEFT 0.20.0. `B` is the global number of prompts per
optimizer update and `R >= 2` is the number of rollouts per prompt. The global
sequence batch is `B * R`. Set gradient accumulation with
`--gradient-accumulation-steps` (default 32). The CLI does not derive or override
it: `B * R = per_device_train_batch_size * world_size * gradient_accumulation_steps` is required.
Set sequences per device with `--per-device-train-batch-size` (default 1).
Inconsistent settings fail before loading model weights. Each rollout batch
has one optimizer update (`num_iterations=1`). LoRA uses rank 16, alpha 32,
zero dropout and all linear layers. The pinned base with adapters disabled is
the fixed reference policy, with KL coefficient 0.05. Advantages are not
normalized. Sampling uses temperature 1.0 without top-k/top-p restriction.
See the [TRL RLOO reference](https://huggingface.co/docs/trl/rloo_trainer).

The optimizer is AdamW (`adamw_torch`). Learning rate is **constant**, with zero
warmup. Complete `checkpoint-N/` directories are saved every **50 optimizer
steps**, and Trainer also saves at the end of training. Each checkpoint contains
LoRA weights, AdamW first/second moments and step counters (`optimizer.pt`), the
constant scheduler state, trainer state/log history, RNG state for every rank,
and the AMP scaler when FP16 is used. Checkpoints also contain
`run_definition.json`, binding them to the pinned model, dataset, training subset,
batch/optimizer settings, precision, world size, and library versions.
The repeat sampler uses `seed + epoch`, so checkpoint resume reconstructs the
same shuffled prompt order, including after an epoch boundary. Checkpoints are
saved between complete rollout/update batches.

Resume a full checkpoint, using the same training settings:

```bash
uv run --locked post-train-math train --model-key olmo2 --output-dir runs/olmo2-rloo --resume-from-checkpoint runs/olmo2-rloo/checkpoint-100 --per-device-train-batch-size 1 --gradient-accumulation-steps 32 --max-steps 200
```

`--max-steps 200` is the **total target**, so this continues from step 100 to
step 200. It may be extended because the constant schedule does not depend on
the original horizon. Incompatible settings and incomplete checkpoints are
rejected before writing run artifacts or loading model weights. `adapter/` is
an inference export; it does not contain AdamW moments and cannot be used for
training resume. An empty new output directory can also resume a checkpoint
from another run; a nonempty directory can only resume its own checkpoint.
`--prepare-only` cannot be combined with resume.

`train_report.json` records `log_history` directly from
`trainer.state.log_history`, `train_metrics` from the final `trainer.train()`
result, the final global step, and the resume checkpoint path. Resumed history
retains the earlier steps loaded from the checkpoint. The final trainer state
is also saved as `trainer_state.json`.

Training starts from exactly the frozen **5,589** `boxed-numeric` train problems.
For each model, it tokenizes every full conversation with that model's actual
chat template, including the fixed system prompt, role/BOS tokens, and assistant
generation prefix. `tokenize=True`, `add_generation_prompt=True`,
`truncation=False`, and `padding=False` are explicit. Only problems with
`prompt_tokens <= 2048` enter RLOO. RLOO also disables truncation and uses the
same conversation and template settings. The underlying cohort and all split
memberships remain fixed; the training subset is specific to each model.
The completion budget remains 2,048 tokens.

Before loading model weights, training saves these files in its output directory:

| File | Contents |
| --- | --- |
| `prompt_audit.jsonl` | All 5,589 problem IDs, full prompt token counts, subject (`type`), difficulty (`level`), selection decisions, and token-ID hashes |
| `overlong_prompts.jsonl` | Excluded problem IDs, token counts, subject, and difficulty; empty if none |
| `training_subset.jsonl` | Every selected record, including exact chat messages, gold rational values, ID, token count, subject, and difficulty |
| `training_subset_manifest.json` | Parent data manifest hash, checkpoint commit, actual chat template and hash, tokenizer state hash, special tokens, library versions, selection rule, counts, and artifact SHA256 hashes |

The source has no native problem ID. `problem_id` is `math-` followed by SHA256
of the original problem text encoded as UTF-8. It is stable across models.
Subset definitions and JSONL records are deterministic for the same source and
tokenizer. The training report also records the subset manifest hash.

Prepare and inspect a subset on CPU, without downloading model weights:

```bash
uv run --locked post-train-math train --model-key olmo2 --prepare-only --output-dir runs/olmo2-subset
uv run --locked post-train-math train --model-key granite --prepare-only --output-dir runs/granite-subset
```

When no local model directory exists, `--prepare-only` fetches only the tokenizer
from the pinned checkpoint. A later training run recomputes and saves its own
subset in a fresh output directory. Test evaluation uses left
truncation policy and does not apply the training-only length filter.

The saved audits in `cohort/training-subsets/` use the pinned checkpoints above:

| Model | Audited | Selected | Overlong | Maximum prompt tokens |
| --- | ---: | ---: | ---: | ---: |
| OLMo2 | 5,589 | 5,589 | 0 | 1,350 |
| Granite | 5,589 | 5,589 | 0 | 1,502 |

Both models currently retain all 5,589 problems. Their overlong JSONL files are
empty; the filter and 2048/2049-token boundary remain enforced.

Training uses seed 42, gradient checkpointing, and BF16 when available, otherwise
FP16. For two GPUs with B=8, R=4, per-device-train-batch-size=1, specify
`--gradient-accumulation-steps 16` with
`accelerate launch --num_processes 2 -m post_train_math.cli train ...`.
Evaluation runs on one GPU.

## Repository

```text
post-train-math/
├── cohort/
│   ├── manifest.json
│   ├── testing-subset/  # fixed 512 original-test questions and sampling audit
│   └── training-subsets/  # saved native-tokenizer audits for both checkpoints
├── src/post_train_math/
│   ├── answers.py
│   ├── data.py
│   ├── models.py
│   ├── rloo.py
│   ├── cli.py
│   ├── testing.py
│   ├── evaluation.py
│   └── curves.py
├── tests/
├── pyproject.toml
├── uv.lock
└── README.md
```

`post_train_math` is a namespace package. Downloads, model weights, run outputs,
virtual environments, and caches are created locally by the commands above.
They are not part of the source tree.

## Checks

```bash
uv lock --check
uv run --locked ruff check src tests
uv run --locked pytest -q
uv build
```

Tests cover the grammar, strict gold-box requirement, reward, original train/test
membership and order, source/file integrity, model pins, explicit batch settings, native
chat token counts, the 2048/2049-token boundary, saved subset hashes, CLI workflows, and a
small real TRL/PEFT CPU training step, adapter export, fresh-model checkpoint
inference reload, fixed test selection, and offline plotting. They do not run the
full GPU experiment or claim any model accuracy. Historical baseline artifacts
and local experiments have been removed.
