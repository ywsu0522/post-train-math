# post-train-math

Pinned Hendrycks MATH data, the frozen `boxed-numeric` cohort, and RLOO / GR-REINFORCE
training with a shared explicit PyTorch + PEFT loop for two pinned checkpoints. Python 3.12 and uv are
required. Data preparation and tests run on CPU; evaluation and training require CUDA.

The experiment studies how different RL methods change mathematical reasoning
under fixed data, prompts, verifier rewards and evaluation settings. `hw4.pdf`
is a reference for RL objectives and update mechanics only. Its model, task,
cohort, prompts and reward design do not define this project's experiment.
There is no hyperparameter search or best-model selection stage; retain the
planned checkpoint evaluations to compare learning trajectories.
GR-REINFORCE uses a fixed KL coefficient of 0.05; RLOO retains its coefficient
of 0.01. Select the method with `train --method rloo` (default) or
`train --method gr_reinforce`; both use the same data, prompts and verifier.

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

# Train GR-REINFORCE with the same batch and rollout budget.
uv run --locked post-train-math train --method gr_reinforce --model-key olmo2 --output-dir runs/olmo2-gr-reinforce --prompts-per-update 8 --rollouts 4 --per-device-train-batch-size 1 --gradient-accumulation-steps 32 --max-steps 200

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

Training defaults to 200 rollout iterations and saves `checkpoint-50`,
`checkpoint-100`, `checkpoint-150`, and `checkpoint-200`, indexed by rollout
iteration. One rollout iteration collects one new batch, with BG sampled
completions. `rollout_iterations` and `optimizer_steps` are recorded independently
in history, training reports, checkpoints and evaluation reports. This RLOO
recipe makes exactly one optimizer step per new rollout batch, so the two
counter values currently coincide; that equality is not a cross-method rule.
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

Both methods use PyTorch + PEFT directly. Accelerate handles device placement, DDP,
autocast and AMP; it does not own the iteration loop. There is no RLOOTrainer
subclass, sampler wrapper, or Trainer callback in the RLOO path.

Read `rl_core.py:train_loop` from top to bottom. Each iteration has three explicit
stages, with a start and completion event printed immediately and appended to
`iteration_log.jsonl`:

| Stage | Function | Observable state |
| --- | --- | --- |
| 1.1 | `generate_rollouts` | Draw B prompts **with replacement**, repeat each G times, generate BG completions with the PEFT policy; log sampled indices and completion count |
| 1.2 | `method.estimate_gradients` | Selected method computes rewards, advantages and objective, then calls the core accumulation tool; returns only when unscaled, audited LoRA `.grad` is ready |
| 2 | `update_parameters` | Warmup LR, gradient clipping, `AdamW.step()`, then actual LoRA parameter differences before clearing gradients |

The loop verifies that every trainable parameter is a LoRA A/B matrix in one
of the seven configured projections and that AdamW contains exactly those
parameters. Stage 1.2 rejects gradients on frozen base parameters and records
per-target gradient norms. Stage 2 records `updated`, `changed_lora_tensors`,
`lora_parameter_delta_norm`, and per-target delta norms. A zero-advantage batch
can legitimately have no parameter change; the log reports that fact.

For extra prints or logging, edit `rl_core.log_stage`, insert statements directly
in `rl_core.train_loop`, or inspect `rollouts`, `advantages`, `logps` and parameter `.grad`
in the stage functions. Reward/advantage arrays and their mean, std, min and max
are saved alongside the gradient diagnostics.

`B` is the global number of prompt draws per rollout iteration; `G >= 2`
(`--rollouts`) is the number of answers per draw. Sequences per device are set
by `--per-device-train-batch-size` (default 1). The explicit accumulation setting
(default 32) must satisfy:

```text
B * G = per_device_train_batch_size * world_size * gradient_accumulation_steps
```

Each rank processes a contiguous slice of the global BG sequences. Rewards are
gathered in prompt-major order before computing the G-way baseline, including
when siblings span ranks. Each microbatch policy loss is a sum divided by the
local rollout count; DDP averages ranks, giving the global BG policy average. Accumulation
is explicit in stage 1.2; Accelerate's accumulation factor stays at one.

The RLOO objective is the on-policy REINFORCE estimator:

```text
shaped_reward = verifier_reward - beta * (policy_log_prob - reference_log_prob)
advantage[i,g] = shaped_reward[i,g] - mean(shaped_reward[i,h] for h != g)
loss = -mean(advantage.detach() * sum_completion_token_log_probs)
```

The shaped reward uses a no-grad policy pass. A second policy forward builds
the LoRA graph for backward, so the full BG batch's graphs need not stay in
memory. This single-update objective has the same on-policy gradient as TRL's
sequence-ratio surrogate at ratio one; there is no PPO reuse or clipping.
Advantages are not standardized and token log probabilities are summed, not
length averaged. LoRA uses rank 16, alpha 32, dropout 0, and explicitly targets
attention `q_proj`, `k_proj`, `v_proj`, `o_proj` and MLP `gate_proj`, `up_proj`,
`down_proj`. The reference is the pinned base with adapters disabled; KL beta
is 0.01. Sampling uses temperature 1, top-p 1 and top-k 0.

The two method modules have the same interface:

```python
rollouts = generate_rollouts(...)
gradient_stats = method.estimate_gradients(context, rollouts)
update_stats = update_parameters(...)
```

`training.py` selects the module and initializes model/LoRA/AdamW/Accelerator.
`rl_core.py` imports neither the method modules nor the training entry point.
The core owns token log-probabilities/masks, explicit microbatch accumulation,
DDP synchronization, AMP unscaling, gradient audits, clipping and optimizer steps.
Methods own their estimator and call the common `accumulate_gradients` tool;
they never step the optimizer. No duplicated method-specific training loop is used.

| Objective choice | RLOO | GR-REINFORCE |
| --- | --- | --- |
| Advantage | Leave-one-out baseline of KL-shaped rewards | Raw verifier rewards, group population std + 1e-6, then clip to [-5, 5]; no second normalization |
| Policy term | Completion-token log-probability sum, then completion mean | Completion-token log-probability mean, then completion mean |
| KL | Detached sequence log-ratio in the reward | Independent differentiable `expm1(delta) - delta` loss, delta = ref minus policy log-probability clamped to [-20, 20] |
| KL coefficient | 0.01 | 0.05 |
| Updates per fresh rollout batch | 1 | 1 |

GR-REINFORCE normalizes KL by the total valid completion-token count across the
whole rollout and all ranks. DDP's rank averaging is compensated explicitly.
This keeps the objective invariant to microbatch/rank partitioning with unequal
completion lengths. It differs from averaging each minibatch's token mean with
equal weight, as in the homework starter's accumulation instructions. The
policy term still weights completions equally. Raw-policy scoring requires
sampling temperature=1, top-p=1 and top-k=0; other values are rejected rather
than silently changing the estimator's sampling distribution.

Groups with zero advantages are retained without resampling or filtering.
Completions that reach the token limit without EOS remain eligible for the same
verifier and policy loss; the first EOS is included when present. Non-finite
losses or gradients and AMP-skipped optimizer steps abort training immediately.
A successful optimizer step is counted even when its parameter delta is zero.

The optimizer is `torch.optim.AdamW`, with betas `(0.9, 0.95)`, epsilon `1e-8`,
weight decay `0`, and gradient norm clipping at `0.5`. For one-based optimizer
step `u`, stage 2 explicitly sets `lr = target_lr * min(u / 20, 1)`: the first update
uses target/20, update 20 reaches the target (default `1e-5`), and later updates
keep it constant. No separate scheduler object or scheduler checkpoint is needed.
Warmup uses the restored optimizer-step counter, independently of rollout
iterations; a future method with two updates per rollout reaches the target LR
on optimizer step 20, after ten rollout iterations.

Complete `checkpoint-N/` directories are saved every 50 iterations and at the
end. They contain the adapter/tokenizer, `optimizer.pt`, `trainer_state.json`,
`run_definition.json`, and each rank's Torch CPU/device RNG state. FP16 scaler
state is included with that rank's RNG state. The sampler uses a separate local
generator seeded with `seed + iteration`, so prompt draws need no epoch cursor.
Checkpoints are taken only after a full rollout/update iteration.
`global_step` remains a legacy alias of the rollout-iteration counter in new
state files; new consumers use the two explicit counters. Older single-update
RLOO checkpoints remain readable via their known one-update-per-rollout mapping.

The shared checkpoint engine is `explicit-rl-v1`. Its run definition records the
method, beta and complete objective settings. Previous `explicit-rloo-v1`
checkpoints are normalized to the equivalent RLOO settings for resume; changing
methods or objectives is rejected. Old TRL checkpoints remain
usable for inference/evaluation but are rejected for resume by this loop: the
sampling rule, LR indexing and checkpoint layout have changed. This avoids
silently treating a different training recipe as an exact continuation.

Resume a full checkpoint, using the same training settings:

```bash
uv run --locked post-train-math train --model-key olmo2 --output-dir runs/olmo2-rloo --resume-from-checkpoint runs/olmo2-rloo/checkpoint-100 --per-device-train-batch-size 1 --gradient-accumulation-steps 32 --max-steps 200
```

`--max-steps 200` is the **total rollout-iteration target**, so this continues
from rollout iteration 100 to 200. Resume restores the adapter, AdamW moments,
RNG state and both progress counters to continue the same interrupted experiment.
It may be extended because the fixed warmup followed by constant LR
does not depend on the original horizon. Incompatible settings and incomplete
checkpoints are rejected before writing run artifacts or loading model weights. `adapter/` is
an inference export; it does not contain AdamW moments and cannot be used for
training resume. An empty new output directory can also resume a checkpoint
from another run; a nonempty directory can only resume its own checkpoint.
`--prepare-only` cannot be combined with resume.

`train_report.json` records the final iteration, explicit-loop settings,
training-subset summary and per-iteration gradient/update history. Resumed
history retains the earlier iterations from `trainer_state.json`. Stage events
continue to append to `iteration_log.jsonl`; the actual checkpoint and final
adapter each carry their own model/data metadata for evaluation.

Training starts from exactly the frozen **5,589** `boxed-numeric` train problems.
For each model, it tokenizes every full conversation with that model's actual
chat template, including the fixed system prompt, role/BOS tokens, and assistant
generation prefix. `tokenize=True`, `add_generation_prompt=True`,
`truncation=False`, and `padding=False` are explicit. Only problems with
`prompt_tokens <= 2048` enter training. Both methods disable truncation and uses the
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
│   ├── rl_core.py  # shared visible loop, rollouts, backward/DDP/AMP and updates
│   ├── rloo.py  # RLOO estimator and objective
│   ├── gr_reinforce.py  # group-normalized REINFORCE estimator and KL objective
│   ├── training.py  # settings, model/optimizer setup, CLI entry
│   ├── training_data.py  # native-chat prompt preparation and audits
│   ├── checkpoints.py  # adapter/AdamW/RNG save and restore
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
small real OLMo2/Granite PEFT CPU training steps for both methods, analytical vs
microbatch gradients, unequal-length global KL/rank normalization, frozen-base checks, actual LoRA updates, warmup/resume, adapter export,
fresh-model checkpoint inference reload, fixed test selection, and offline plotting. They do not run the
full GPU experiment or claim any model accuracy. Historical baseline artifacts
and local experiments have been removed.
