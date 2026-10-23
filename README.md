# post-train-math

Pinned Hendrycks MATH data, the frozen `boxed-numeric` cohort, and RLOO
training with TRL + PEFT for two pinned checkpoints. Python 3.12 and uv are
required. Data preparation and tests run on CPU; evaluation and training require CUDA.

## Frozen data contract

The order is fixed in code and `cohort/manifest.json`:

1. Verify the original source files and their SHA256 hashes.
2. Split **all 7,500 original train problems** into **6,750 train / 750 dev**
   with seed 42, dev ratio 0.1, and type/level stratification.
3. Keep the **original 5,000 test problems unchanged**, including their order.
4. Apply the **same** `boxed-numeric` rule independently within train, dev,
   and test. Never filter before splitting or re-split the eligible subset.

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
| train | 6,750 | 5,029 |
| dev | 750 | 560 |
| test | 5,000 | 3,686 |

Processed Parquet files retain every original row plus eligibility and exact
gold metadata. Consumers select `numeric_eligible` within the requested split.
Preparation records provenance, counts, exclusions, and file hashes. Evaluation
and training reject files that differ from the frozen hashes, even if their
local manifest was edited to match. This preserves the frozen problem groups.

Prediction reward uses only the final boxed marker: 1 for exact rational equality,
0 otherwise. A malformed final box cannot fall back to an earlier box.

## Four workflows

```bash
uv sync --locked --group dev

# 1. Download, split, and annotate the data.
uv run --locked post-train-math data download
uv run --locked post-train-math data prepare

# 2. Download the pinned model checkpoints.
uv run --locked post-train-math model download --model-key olmo2
uv run --locked post-train-math model download --model-key granite

# 3. Evaluate the dev cohort with greedy decoding.
uv run --locked post-train-math eval --model-key olmo2 --output-dir runs/olmo2-dev
uv run --locked post-train-math eval --model-key granite --output-dir runs/granite-dev

# 4. Train a LoRA adapter with RLOO.
uv run --locked post-train-math train --model-key olmo2 --output-dir runs/olmo2-rloo --prompts-per-update 8 --rollouts 4 --gradient-accumulation-steps 32 --max-steps 100

# Evaluate the trained adapter.
uv run --locked post-train-math eval --model-key olmo2 --adapter-dir runs/olmo2-rloo/adapter --output-dir runs/olmo2-rloo-dev
```

Use `--split test` for test evaluation, and `--limit N` for a recorded smoke
evaluation. Model/data directories may be overridden with `--model-dir` and
`--data-dir`. Run output directories must be empty. The module entry point is
`python -m post_train_math.cli`.

| Key | Checkpoint | Commit |
| --- | --- | --- |
| `olmo2` | `allenai/OLMo-2-0425-1B-SFT` | `0d85a3d037876ce6ac7d4311d994400fc66ac27f` |
| `granite` | `ibm-granite/granite-3.3-2b-instruct` | `707f574c62054322f6b5b04b6d075f0a8f05e0f0` |

Both models use their native chat templates and explicit fixed system prompts.
Training reads only the train cohort. Dev and test are used by the evaluation
workflow. Training saves `adapter/`, `train_report.json`, and the subset audit below; evaluation saves
`predictions.jsonl` and `eval_report.json`.

RLOO uses TRL 1.12.0 and PEFT 0.20.0. `B` is the global number of prompts per
optimizer update and `R >= 2` is the number of rollouts per prompt. The global
sequence batch is `B * R`. Set gradient accumulation with
`--gradient-accumulation-steps` (default 32). The CLI does not derive or override
it: `B * R = micro_batch * world_size * gradient_accumulation_steps` is required.
Inconsistent settings fail before loading model weights. Each rollout batch
has one optimizer update (`num_iterations=1`). LoRA uses rank 16, alpha 32,
zero dropout and all linear layers. The pinned base with adapters disabled is
the fixed reference policy, with KL coefficient 0.05. Advantages are not
normalized. Sampling uses temperature 1.0 without top-k/top-p restriction.
See the [TRL RLOO reference](https://huggingface.co/docs/trl/rloo_trainer).

Training starts from exactly the frozen **5,029** `boxed-numeric` train problems.
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
| `prompt_audit.jsonl` | All 5,029 problem IDs, full prompt token counts, subject (`type`), difficulty (`level`), selection decisions, and token-ID hashes |
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
subset in a fresh output directory. Dev/test evaluation keeps its existing left
truncation policy and does not apply the training-only length filter.

The saved audits in `cohort/training-subsets/` use the pinned checkpoints above:

| Model | Audited | Selected | Overlong | Maximum prompt tokens |
| --- | ---: | ---: | ---: | ---: |
| OLMo2 | 5,029 | 5,029 | 0 | 1,350 |
| Granite | 5,029 | 5,029 | 0 | 1,502 |

Both models currently retain all 5,029 problems. Their overlong JSONL files are
empty; the filter and 2048/2049-token boundary remain enforced.

Training uses seed 42, gradient checkpointing, and BF16 when available, otherwise
FP16. For two GPUs with B=8, R=4, micro-batch=1, specify
`--gradient-accumulation-steps 16` with
`accelerate launch --num_processes 2 -m post_train_math.cli train ...`.
Evaluation runs on one GPU.

## Repository

```text
post-train-math/
├── cohort/
│   ├── manifest.json
│   └── training-subsets/  # saved native-tokenizer audits for both checkpoints
├── src/post_train_math/
│   ├── answers.py
│   ├── data.py
│   ├── models.py
│   ├── rloo.py
│   └── cli.py
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

Tests cover the grammar, strict gold-box requirement, reward, split-before-filter
membership, source/file integrity, model pins, explicit batch settings, native
chat token counts, the 2048/2049-token boundary, saved subset hashes, all four CLI workflows, and a
small real TRL/PEFT CPU training step with adapter export. They do not run the
full GPU experiment or claim any model accuracy. Historical baseline artifacts
and local experiments have been removed.
