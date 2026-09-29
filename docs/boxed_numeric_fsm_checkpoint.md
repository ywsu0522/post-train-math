# Boxed numeric FSM checkpoint

This checkpoint records the current handoff state on branch `boxed-numeric-v1`.
The working tree contains the earlier `boxed-numeric-v1` contract migration plus
the prompt/FSM experiment scaffolding described here. No model experiment has
been run in this environment because it has CPU-only Torch (`2.13.0+cpu`), no
CUDA device, and no Base or SFT checkpoint under `models/` or `runs/`.

## Contract decisions

- Every processed split is consumed through `numeric_eligible == True`. SFT,
  evaluation, RL dataset construction, and the matrix runner all use this
  subset.
- Base, SFT, and RL use the single `boxed` prompt formatter. It serializes the
  same fixed `SYSTEM_PROMPT` followed by `Problem:` and `Solution:`. The prompt
  contract and SHA256 are recorded in run metadata.
- The FSM is syntax-only. It detects literal `\\boxed{` in generated tokens,
  masks token candidates that would violate `number | fraction`, and permits
  the outer `}` only from an accepting state. After the outer `}` it returns to
  free generation; additional boxes are allowed and independently constrained.
- FSM does not inspect canonical values, reject zero denominators, force a box,
  or terminate generation. The existing verifier remains responsible for
  selecting the final literal box and comparing canonical `Fraction` values.
- The tokenizer adapter currently requires a fast tokenizer with a ByteLevel
  decoder. This matches the pinned OLMo tokenizer. Unsupported decoders fail
  explicitly instead of silently applying an incorrect byte mapping.

## Implemented entry points

- `posttrain_math.fsm`: byte-level transition machine and
  `BoxedNumericLogitsProcessor`.
- `HFModelRunner.from_pretrained(..., fsm=False)` and `HFModelRunner(..., fsm=...)`.
  Evaluation exposes `posttrain-math eval --fsm off|on`.
- `posttrain-math eval-format-matrix --sft-model ... --output-dir ...` runs, in
  order, Base/FSM-off, Base/FSM-on, SFT/FSM-off, SFT/FSM-on with identical
  split, prompt, generation settings, and optional limit. It writes one
  `matrix.json` plus one metrics directory per condition.
- Evaluation metrics now include `grammar_output_rate`, alongside boxed and
  numeric output rates. This measures syntax independently of reward.

## Remaining work for the next agent

1. Run `uv run --locked --no-sync posttrain-math data prepare` if the processed
   data directory is stale, then provision the pinned Base model and an SFT
   checkpoint on a CUDA host.
2. Run the four matrix conditions with the same `--split`, `--limit`,
   `--batch-size`, and `--max-new-tokens`; archive `matrix.json`, metrics, and
   generation outputs.
3. Review real tokenizer behavior on the pinned model before a long run. In
   particular, verify token bytes for `\\boxed{`, `\\frac{`, `\\%`, and `}` and
   inspect a few masked distributions. The unit tests use a synthetic ByteLevel
   tokenizer and a tiny GPT-2 generation test.
4. If the RLVR rollout path must also have an FSM-on condition, thread the same
   `LogitsProcessor` semantics into the TRL generation hook. The current FSM
   integration is complete for the evaluation runner and the four requested
   Base/SFT format comparisons; `train_rl` itself still uses TRL's generation
   path without this processor.
5. Decide whether to add a persistent `fsm` field to RL run configuration once
   the TRL integration point is selected. Do not change the verifier contract.

## Validation at checkpoint

With `PYTHONPATH=src` (the local editable `.pth` is unreadable in this Windows
environment), these checks pass:

```text
uv run --locked --no-sync pytest -q
153 passed, 1 warning (pytest cache permission warning)

uv run --locked --no-sync ruff check .
All checks passed
```

The earlier boxed-numeric data migration and pinned raw dataset audit remain in
`data/processed/manifest.json`; they reported 5,586/7,500 eligible train rows,
3,686/5,000 eligible test rows, and 9,272/12,500 combined.
