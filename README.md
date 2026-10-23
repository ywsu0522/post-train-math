# post-train-math

This repository records two completed pieces of work:

1. A pinned Hendrycks MATH source, deterministic train/dev split, and the
   `boxed-numeric-v2` research cohort.
2. The original dev baseline measurements for OLMo2-1B-SFT and Granite-3.3-2B-IFT.

Training is not part of the current implementation. The next experiment is RLOO
with `max_prompt_length = max_completion_length = 2048`, B prompts per update,
and R rollouts per prompt. Those runs have not been completed.

## Fixed dataset and cohort

Source: `DigitalLearningGmbH/MATH-lighteval` at
`f06834690385b29df31ccc717250746a3ba0322b` (the consolidated Hendrycks MATH data).
The original 7,500 training problems are split into 6,750 train and 750 dev
problems using seed 42, dev ratio 0.1, and type/level stratification. The original
5,000 test problems remain separate. Cohort filtering preserves this split and
includes all levels and subjects.

Pinned source file SHA256:

| File | SHA256 |
| --- | --- |
| `data/train-00000-of-00001.parquet` | `eca6e667f4305dd5e5ba09b4fd55e7f3174a0fbe361cdfd4c44758b593a76933` |
| `data/test-00000-of-00001.parquet` | `7dca8d6e41af88ecf82f2b5f36eb5530e083aaaa86ee325f62bd5c31535178c6` |

```text
digits         := [0-9]+
number         := ["-"] digits ["." digits] ["\%"]
fraction_latex := ["-"] "\frac{" digits "}{" digits "}"
fraction_slash := ["-"] digits "/" digits
answer         := number | fraction_latex | fraction_slash
```

A gold solution must contain exactly one literal `\boxed` marker, with balanced,
nonempty braces. After stripping outer whitespace, its content must fully match
this grammar. Internal whitespace, plus signs, signed denominators, and zero
denominators are rejected. Percentages are divided by 100; all accepted forms
are compared as exact rational values using `Decimal` and `Fraction`.

Prediction scoring examines the final boxed marker. A malformed final marker
does not fall back to an earlier box. No symbolic verifier or constrained decoding
is used. The grammar and extraction rules are implemented in
[`answers.py`](src/post_train_math/answers.py).

Preparation verifies source hashes and records the source identity, processed
file hashes, split settings, cohort counts, and exclusions in `manifest.json`.
The verified current snapshot is committed as [`cohort/manifest.json`](cohort/manifest.json):
5,029 train, 560 dev, and 3,686 test problems are eligible.
Evaluation verifies that manifest and the processed files before loading a model.

## Completed baseline measurements

| Candidate | Correct / dev cohort | Accuracy | Completion budget |
| --- | --- | --- | --- |
| OLMo2-1B-SFT | 101 / 544 | 18.57% | 1024 |
| Granite-3.3-2B-IFT | 344 / 544 | 63.24% | 1024 |

The original ZIPs, including predictions, metrics, and reports, are stored in
[`baselines/`](baselines/README.md), with checksums in
[`baselines/manifest.json`](baselines/manifest.json).

These measurements used **exact-rational-v1**, greedy decoding, and each model's
native chat template with its recorded system prompt. They are historical
measurements, not measurements on the current v2 cohort or at a 2048-token
budget. They have not been relabeled or replaced. New v2 evaluations use separate
output directories. The candidate revisions and original prompt text remain
pinned in [`baselines.py`](src/post_train_math/baselines.py).

## Usage

The repository, package distribution, and CLI are named `post-train-math`.
The Python package is `post_train_math` (Python identifiers use underscores):
`python -m post_train_math` is equivalent to the CLI.

Python 3.12 and uv are required. Model evaluation requires CUDA. Data preparation
and unit tests can run on CPU.

```bash
uv sync --locked --group dev
uv run --locked post-train-math data download
uv run --locked post-train-math data prepare
uv run --locked post-train-math model download --model-key olmo2
uv run --locked post-train-math model download --model-key granite
uv run --locked post-train-math eval --model-key olmo2 --output-dir runs/olmo2-v2-dev
uv run --locked post-train-math eval --model-key granite --output-dir runs/granite-v2-dev
```

The evaluation commands above run the current v2 cohort with the retained
1024-token default. They do not recreate the old 544-question cohort. Use
`--max-new-tokens` explicitly for a new generation budget. `--limit` is available
for a smoke evaluation and is recorded in the output.

For Linux GPU environments, `bash scripts/launch_gpu.sh 2 eval --model-key granite
--output-dir runs/granite-v2-dev-2gpu` uses two processes and merges the predictions.
Each process has its own model copy. Output directories must be empty.

If existing raw files fail source verification, use a new location:

```bash
uv run --locked post-train-math data download --output-dir data/pinned-raw
uv run --locked post-train-math data prepare --raw-dir data/pinned-raw --output-dir data/pinned-processed
```

Pass `--data-dir data/pinned-processed` to evaluation when using that location.
Downloaded data, model weights, and new run outputs are gitignored.

## Checks

```bash
uv lock --check
uv run --locked ruff check src tests
uv run --locked pytest -q
```
