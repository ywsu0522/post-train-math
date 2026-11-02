"""GR-REINFORCE gradient estimator with group-relative rewards and KL loss."""

from __future__ import annotations

import torch

from post_train_math.answers import numeric_reward
from post_train_math.rl_core import (
    GradientContext,
    LossTerms,
    PreparedRollouts,
    Rollouts,
    accumulate_gradients,
    completion_log_probs,
    log_advantages,
    score_rollouts,
)

NAME = "gr_reinforce"
DEFAULT_BETA = 0.05
DEFAULT_NUM_ITERATIONS = 1
ADVANTAGE_EPS = 1e-6
ADVANTAGE_CLIP = 5.0
LOG_RATIO_CLIP = 20.0
OBJECTIVE = {
    "advantage": "group_population_std",
    "advantage_eps": ADVANTAGE_EPS,
    "advantage_clip": ADVANTAGE_CLIP,
    "normalize_advantages_again": False,
    "policy_reduction": "sequence_mean_then_completion_mean",
    "kl": "differentiable_expm1_log_ratio",
    "kl_log_ratio_clip": LOG_RATIO_CLIP,
    "kl_reduction": "global_rollout_token_mean",
}


def group_relative_advantages(rewards, group_size):
    if group_size < 2 or rewards.numel() % group_size:
        raise ValueError("GR-REINFORCE needs complete groups with G >= 2")
    grouped = rewards.reshape(-1, group_size)
    advantages = (grouped - grouped.mean(1, keepdim=True)) / (
        grouped.std(1, keepdim=True, unbiased=False) + ADVANTAGE_EPS
    )
    return advantages.reshape(-1).clamp(-ADVANTAGE_CLIP, ADVANTAGE_CLIP).detach()


def approximate_kl(logps, ref_logps, token_weight):
    """Shared masked KL contribution; token_weight includes DDP compensation."""
    delta = (ref_logps.detach() - logps.values).clamp(-LOG_RATIO_CLIP, LOG_RATIO_CLIP)
    return ((torch.expm1(delta) - delta) * logps.mask).sum() * token_weight


def loss_terms(logps, ref_logps, advantages, *, sequence_count, token_weight, beta):
    """Normalized local contributions; DDP rank averaging recovers the objective."""
    policy_loss = -(advantages.detach() * logps.means).sum() / sequence_count
    kl = approximate_kl(logps, ref_logps, token_weight)
    return LossTerms(
        policy_loss + beta * kl,
        {"policy_loss": policy_loss, "approx_kl": kl, "kl_loss": beta * kl},
    )


def prepare_rollouts(
    context: GradientContext,
    rollouts: Rollouts,
    *,
    reward_fn=numeric_reward,
    cache_old_log_probs=False,
):
    """Prepare the same group advantages and reference cache for both GR methods."""
    accelerator, config = context.accelerator, context.config
    policy = accelerator.unwrap_model(context.model)
    count = len(rollouts.records)
    micro = config.per_device_train_batch_size
    rewards = accelerator.gather(score_rollouts(context, rollouts, reward_fn))
    advantages = group_relative_advantages(rewards, config.num_generations)
    local = context.local_advantages(advantages, count)
    log_advantages(context, advantages, verifier_rewards=rewards.tolist())
    policy.eval()
    reference = []
    old = []
    for start in range(0, count, micro):
        with torch.no_grad(), accelerator.autocast():
            if cache_old_log_probs:
                old.append(
                    completion_log_probs(
                        policy,
                        context.tokenizer,
                        rollouts,
                        start,
                        start + micro,
                        accelerator.device,
                    ).values.detach()
                )
            with policy.disable_adapter():
                reference.append(
                    completion_log_probs(
                        policy,
                        context.tokenizer,
                        rollouts,
                        start,
                        start + micro,
                        accelerator.device,
                    ).values.detach()
                )
    # DDP averages gradients across ranks. Compensate once so token means are
    # global even when ranks/microbatches contain different completion lengths.
    total_tokens = accelerator.reduce(
        torch.tensor(
            sum(map(len, rollouts.completion_ids)),
            device=accelerator.device,
            dtype=torch.float64,
        ),
        reduction="sum",
    ).item()
    if total_tokens <= 0:
        raise ValueError("Group-relative methods need nonempty completion tokens")
    return PreparedRollouts(
        advantages=local,
        reward_mean=rewards.mean().item(),
        reference_log_probs=reference,
        old_log_probs=old,
        token_weight=accelerator.num_processes / total_tokens,
    )


def estimate_gradients(
    context: GradientContext,
    rollouts: Rollouts,
    *,
    reward_fn=numeric_reward,
    prepared=None,
):
    """Populate LoRA .grad once per fresh rollout; reference/advantages stay fixed."""
    if prepared is None:
        prepared = prepare_rollouts(context, rollouts, reward_fn=reward_fn)
    count = len(rollouts.records)
    micro = context.config.per_device_train_batch_size

    def loss_fn(logps, start, end):
        return loss_terms(
            logps,
            prepared.reference_log_probs[start // micro],
            prepared.advantages[start:end],
            sequence_count=count,
            token_weight=prepared.token_weight,
            beta=context.config.beta,
        )

    stats = accumulate_gradients(context, rollouts, loss_fn)
    stats["reward_mean"] = prepared.reward_mean
    stats["reference_log_probs_require_grad"] = any(
        p.requires_grad for p in prepared.reference_log_probs
    )
    return stats
