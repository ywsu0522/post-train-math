"""RLOO gradient estimator: KL-shaped rewards, LOO baseline, sequence-sum loss."""

from __future__ import annotations

import torch

from post_train_math.answers import numeric_reward
from post_train_math.rl_core import (
    GradientContext,
    LossTerms,
    Rollouts,
    accumulate_gradients,
    log_advantages,
    score_rollouts,
    sequence_log_probs,
)

NAME = "rloo"
DEFAULT_BETA = 0.01
OBJECTIVE = {
    "advantage": "leave_one_out",
    "policy_reduction": "sequence_sum_then_completion_mean",
    "kl": "detached_sequence_log_ratio_reward_shaping",
}


def leave_one_out_advantages(rewards, group_size):
    if group_size < 2 or rewards.numel() % group_size:
        raise ValueError("RLOO needs complete groups with G >= 2")
    grouped = rewards.reshape(-1, group_size)
    baseline = (grouped.sum(1, keepdim=True) - grouped) / (group_size - 1)
    return (grouped - baseline).reshape(-1).detach()


def estimate_gradients(
    context: GradientContext, rollouts: Rollouts, *, reward_fn=numeric_reward
):
    """Populate LoRA .grad using the original on-policy RLOO estimator."""
    accelerator, config = context.accelerator, context.config
    policy = accelerator.unwrap_model(context.model)
    count = len(rollouts.records)
    micro = config.per_device_train_batch_size
    raw_rewards = score_rollouts(context, rollouts, reward_fn)
    policy.eval()
    policy_logps, ref_logps = [], []
    for start in range(0, count, micro):
        with torch.no_grad(), accelerator.autocast():
            policy_logps.append(
                sequence_log_probs(
                    policy,
                    context.tokenizer,
                    rollouts,
                    start,
                    start + micro,
                    accelerator.device,
                )
            )
            with policy.disable_adapter():
                ref_logps.append(
                    sequence_log_probs(
                        policy,
                        context.tokenizer,
                        rollouts,
                        start,
                        start + micro,
                        accelerator.device,
                    )
                )
    scores = accelerator.gather(
        torch.stack([raw_rewards, torch.cat(policy_logps), torch.cat(ref_logps)], dim=1)
    )
    kl = scores[:, 1] - scores[:, 2]
    rewards = scores[:, 0] - config.beta * kl
    advantages = leave_one_out_advantages(rewards, config.num_generations)
    log_advantages(
        context,
        advantages,
        verifier_rewards=scores[:, 0].tolist(),
        sequence_kl=kl.tolist(),
        policy_log_probs_for_kl=scores[:, 1].tolist(),
        reference_log_probs=scores[:, 2].tolist(),
        shaped_rewards=rewards.tolist(),
    )
    local = context.local_advantages(advantages, count)

    def loss_fn(logps, start, end):
        return LossTerms(-(local[start:end] * logps.sums).sum() / count)

    stats = accumulate_gradients(context, rollouts, loss_fn)
    stats["reward_mean"] = scores[:, 0].mean().item()
    stats["reference_log_probs_require_grad"] = any(p.requires_grad for p in ref_logps)
    return stats
