"""GRPO estimator: fixed rollout statistics and token-level clipped surrogate."""

from __future__ import annotations

import torch

from post_train_math import gr_reinforce
from post_train_math.answers import numeric_reward
from post_train_math.rl_core import (
    GradientContext,
    LossTerms,
    Rollouts,
    accumulate_gradients,
)

NAME = "grpo"
DEFAULT_BETA = 0.05
DEFAULT_NUM_ITERATIONS = 2
CLIP_EPS = 0.2
LOG_RATIO_CLIP = 20.0
OBJECTIVE = {
    **gr_reinforce.OBJECTIVE,
    "policy_objective": "ppo_clipped_surrogate",
    "importance_sampling_level": "token",
    "clip_eps": CLIP_EPS,
    "policy_log_ratio_clip": LOG_RATIO_CLIP,
    "kl_importance_weighting": False,
    "old_policy": "fixed_for_all_updates_on_rollout",
}


def prepare_rollouts(context, rollouts, *, reward_fn=numeric_reward):
    return gr_reinforce.prepare_rollouts(
        context, rollouts, reward_fn=reward_fn, cache_old_log_probs=True
    )


def loss_terms(
    logps, old_logps, ref_logps, advantages, *, sequence_count, token_weight, beta
):
    """Sequence-mean PPO objective plus the same global-token KL as GR-REINFORCE."""
    log_ratio = (logps.values - old_logps.detach()).clamp(
        -LOG_RATIO_CLIP, LOG_RATIO_CLIP
    )
    ratio = log_ratio.exp()
    adv = advantages.detach().unsqueeze(1)
    clipped = ratio.clamp(1 - CLIP_EPS, 1 + CLIP_EPS)
    surrogate = torch.minimum(ratio * adv, clipped * adv)
    policy_loss = (
        -((surrogate * logps.mask).sum(1) / logps.lengths.clamp_min(1)).sum()
        / sequence_count
    )
    kl = gr_reinforce.approximate_kl(logps, ref_logps, token_weight)
    # clipfrac counts tokens outside the interval, irrespective of advantage sign.
    clipfrac = (
        ((ratio < 1 - CLIP_EPS) | (ratio > 1 + CLIP_EPS)) * logps.mask
    ).sum() * token_weight
    return LossTerms(
        policy_loss + beta * kl,
        {
            "policy_loss": policy_loss,
            "approx_kl": kl,
            "kl_loss": beta * kl,
            "clipfrac": clipfrac,
            "ratio_mean": (ratio * logps.mask).sum() * token_weight,
        },
    )


def estimate_gradients(
    context: GradientContext,
    rollouts: Rollouts,
    *,
    reward_fn=numeric_reward,
    prepared=None,
):
    """Populate fresh LoRA gradients; the core updates parameters between passes."""
    if prepared is None:
        prepared = prepare_rollouts(context, rollouts, reward_fn=reward_fn)
    count = len(rollouts.records)
    micro = context.config.per_device_train_batch_size

    def loss_fn(logps, start, end):
        return loss_terms(
            logps,
            prepared.old_log_probs[start // micro],
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
    stats["old_log_probs_require_grad"] = any(
        p.requires_grad for p in prepared.old_log_probs
    )
    return stats
