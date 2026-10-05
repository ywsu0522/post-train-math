# RLVR research contract

This project treats mathematical RLVR as **outcome optimization with a deterministic terminal verifier**. The primary reward is `boxed-numeric-v1`: the generated final boxed number or LaTeX fraction is canonicalized to a `Fraction` and compared with gold for binary reward (`1` or `0`). Numbers include decimals and escaped percentages; invalid final boxes receive `0`. No intermediate reasoning step is graded during training.

That separation is intentional. The project asks two different questions:

1. **Optimization:** under sparse binary rewards, when do different policy-gradient estimators produce useful learning signal efficiently?
2. **Reasoning-path diagnostics:** when terminal reward improves, do sampled textual prefixes move into states from which future success is more likely?

The second question is deliberately narrower than mechanistic interpretability. A high-value textual prefix does not prove that the visible chain of thought is a faithful account of the model's internal computation, and it does not by itself prove that every preceding mathematical step is valid.

## Training backend policy

The [first T4 pilot](rloo_pilot.md) uses RLOO from the original one-pass SFT adapter,
on all eligible MATH training levels/types. It measures actual reward sparsity and
before/after development accuracy. There is no format-rate admission threshold,
continued-SFT stage, or reward-based curriculum selection. The matrix below defines
later estimator comparisons, not three runs required before the first result.

Full 1B training should use maintained trainer implementations rather than project-specific copies of rollout, distributed, optimizer, checkpoint, and mixed-precision machinery. Small reference functions in `posttrain_math.estimators` encode the mathematical semantics we want to verify.

The controlled first-stage matrix is:

| Method | Training backend | Explicit contract |
| --- | --- | --- |
| REINFORCE | reference objective only for now | no baseline, `A=R` |
| RLOO | TRL `RLOOTrainer` | leave-one-out baseline, no advantage normalization, one on-policy iteration |
| GRPO | TRL `GRPOTrainer` | group mean baseline, group-std scaling, original sequence-normalized GRPO loss |
| Dr.GRPO | TRL `GRPOTrainer` | group mean baseline, no group-std scaling, `dr_grpo` loss normalization |
| PPO | reference clipped objective only for now | full production backend deferred until it can use the same verifier/data contract without inflating scope |

For the controlled runs, `num_iterations=1`, `beta=0`, `temperature=1.0`, `top_p=1.0`, `top_k=0`, and dropout is disabled. Equal optimizer-step counts are not the primary fairness criterion; comparisons should report generated completions/tokens, verifier calls, and GPU time.

## Executable estimator references

For one reward group `R_1, ..., R_G`:

- REINFORCE: `A_i = R_i`.
- RLOO: `A_i = R_i - mean(R_{-i})`.
- GRPO: `A_i = (R_i - mean(R)) / (std(R) + 1e-4)` using the same sample-standard-deviation convention as pinned TRL.
- Dr.GRPO: `A_i = R_i - mean(R)`; the trainer additionally uses the Dr.GRPO loss normalization.

For binary reward with per-prompt success probability `p`, a group has at least one success and one failure with probability

`P(mixed) = 1 - p^G - (1-p)^G`.

All-zero and all-one groups have zero group-relative advantage signal. This formula should be compared with observed mixed-group/nonzero-advantage rates before adding dynamic-sampling mechanisms.

## Continuation-value probe

The diagnostic state is a **textual prefix** of a sampled completion. For prompt `x` and completion prefix `y_{1:t}`, define

`V_hat(x, y_{1:t}) = mean_m R(x, y_{1:t} + continuation_m)`

where fresh continuations are sampled from the frozen policy and scored only by `boxed-numeric-v1` terminal reward.

Interpretation: `V_hat` estimates how likely the current policy is to reach a correct terminal answer from that textual state. It is not a trained critic, process reward, symbolic proof checker, or hidden-state explanation.

The first probe intentionally uses only fixed 0%, 25%, 50%, and 75% prefixes of the pre-box textual span. Surface reflection markers such as `wait` or `actually` are not primary event definitions. Event-specific analysis is postponed until the fixed-prefix pilot shows a measurable signal.

### Step 1: freeze frontier prompts with the SFT policy

```bash
uv run --locked --no-sync python -m posttrain_math.reasoning_probe prompt-success \
  --model runs/olmo2-1b-lora-sft-v1/final-model \
  --split dev \
  --rollouts 8 \
  --limit-prompts 128 \
  --output-dir runs/probes/sft-dev-prompt-success
```

This writes raw completions plus `prompt_success.parquet` with `p_hat` and a Wilson interval. The prompt IDs become a frozen selection artifact. `probe_config.json` records the sampling contract and runtime provenance; the prefix-value probe validates that contract and records SHA256 hashes for both selection files before using them.

### Step 2: estimate prefix continuation value on the same frontier prompts

```bash
uv run --locked --no-sync python -m posttrain_math.reasoning_probe prefix-value \
  --model runs/olmo2-1b-grpo-v1/final-model \
  --split dev \
  --selection runs/probes/sft-dev-prompt-success/prompt_success.parquet \
  --min-p 0.125 \
  --max-p 0.875 \
  --limit-prompts 32 \
  --branch-rollouts 8 \
  --output-dir runs/probes/grpo-dev-prefix-value
```

Run the same command with the frozen SFT checkpoint to obtain the first SFT-vs-GRPO comparison. The output keeps the original trajectories, per-prefix estimates, and every branch completion/reward so plots can be audited.

## Deferred scope

Do not add the following before the first controlled results justify them:

- PRM or LLM-as-a-judge process rewards;
- keyword-defined "aha moments" as a primary metric;
- hidden-state/mechanistic interpretability probes;
- custom Triton kernels;
- a second large RL framework solely to obtain PPO;
- DAPO Dynamic Sampling before the all-zero/all-one group dead zone is measured.
