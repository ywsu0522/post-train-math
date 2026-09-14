from __future__ import annotations

import math
from collections.abc import Sequence

import torch


def _reward_tensor(rewards: Sequence[float] | torch.Tensor) -> torch.Tensor:
    if isinstance(rewards, torch.Tensor):
        tensor = rewards
        if not tensor.is_floating_point():
            tensor = tensor.float()
        return tensor
    return torch.tensor(list(rewards), dtype=torch.float32)


def reinforce_advantages(
    rewards: Sequence[float] | torch.Tensor,
) -> torch.Tensor:
    """No-baseline Monte Carlo REINFORCE advantages: A_i = R_i."""
    return _reward_tensor(rewards).clone()


def rloo_advantages(
    rewards: Sequence[float] | torch.Tensor,
) -> torch.Tensor:
    """Leave-one-out sampled baseline used by RLOO for one prompt group."""
    values = _reward_tensor(rewards)
    if values.ndim != 1:
        raise ValueError("RLOO reference expects one 1D prompt group.")
    if values.numel() < 2:
        raise ValueError("RLOO needs at least two completions per prompt.")
    baseline = (values.sum() - values) / (values.numel() - 1)
    return values - baseline


def grpo_advantages(
    rewards: Sequence[float] | torch.Tensor,
    *,
    epsilon: float = 1e-4,
) -> torch.Tensor:
    """TRL-compatible group-standardized GRPO advantages for one prompt group.

    TRL 1.12.0 uses a sample standard deviation (Bessel correction) and adds
    1e-4 to the denominator. This tiny reference exists to make the configured
    trainer semantics executable and unit-testable; it is not a training loop.
    """
    values = _reward_tensor(rewards)
    if values.ndim != 1:
        raise ValueError("GRPO reference expects one 1D prompt group.")
    if values.numel() < 2:
        raise ValueError("GRPO needs at least two completions per prompt.")
    centered = values - values.mean()
    std = values.std(correction=1)
    return centered / (std + epsilon)


def dr_grpo_advantages(
    rewards: Sequence[float] | torch.Tensor,
) -> torch.Tensor:
    """Dr.GRPO reward centering without within-group standard-deviation scaling."""
    values = _reward_tensor(rewards)
    if values.ndim != 1:
        raise ValueError("Dr.GRPO reference expects one 1D prompt group.")
    if values.numel() < 2:
        raise ValueError("Dr.GRPO needs at least two completions per prompt.")
    return values - values.mean()


def reinforce_sequence_loss(
    sequence_log_probs: torch.Tensor,
    rewards: Sequence[float] | torch.Tensor,
) -> torch.Tensor:
    """Educational no-baseline REINFORCE objective over sequence log-probability."""
    reward_values = _reward_tensor(rewards).to(
        device=sequence_log_probs.device,
        dtype=sequence_log_probs.dtype,
    )
    if sequence_log_probs.ndim != 1 or reward_values.ndim != 1:
        raise ValueError("Expected 1D sequence log-probabilities and rewards.")
    if sequence_log_probs.shape != reward_values.shape:
        raise ValueError("sequence_log_probs and rewards must have equal shape.")
    return -(reward_values * sequence_log_probs).mean()


def ppo_clipped_surrogate_loss(
    log_ratio: torch.Tensor,
    advantages: torch.Tensor,
    *,
    epsilon: float = 0.2,
) -> torch.Tensor:
    """Small reference for the PPO clipped surrogate objective, not a PPO trainer."""
    if log_ratio.shape != advantages.shape:
        raise ValueError("log_ratio and advantages must have equal shape.")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive.")
    ratio = torch.exp(log_ratio)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * advantages
    return -torch.minimum(unclipped, clipped).mean()


def mixed_group_probability(success_probability: float, group_size: int) -> float:
    """P(group contains both reward 0 and reward 1) for iid binary rollouts."""
    p = float(success_probability)
    if not 0.0 <= p <= 1.0:
        raise ValueError("success_probability must be in [0, 1].")
    if group_size < 2:
        raise ValueError("group_size must be at least 2.")
    return 1.0 - math.pow(p, group_size) - math.pow(1.0 - p, group_size)
