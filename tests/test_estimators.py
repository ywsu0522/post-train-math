import pytest
import torch

from posttrain_math.estimators import (
    dr_grpo_advantages,
    grpo_advantages,
    mixed_group_probability,
    ppo_clipped_surrogate_loss,
    reinforce_sequence_loss,
    rloo_advantages,
)


def test_rloo_leave_one_out_baseline() -> None:
    actual = rloo_advantages([1.0, 0.0, 0.0, 0.0])
    expected = torch.tensor([1.0, -1.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0])
    torch.testing.assert_close(actual, expected)
    assert actual.mean().item() == pytest.approx(0.0, abs=1e-7)


def test_grpo_and_dr_grpo_have_distinct_scaling() -> None:
    rewards = [1.0, 0.0, 0.0, 0.0]
    dr = dr_grpo_advantages(rewards)
    grpo = grpo_advantages(rewards)

    torch.testing.assert_close(
        dr,
        torch.tensor([0.75, -0.25, -0.25, -0.25]),
    )
    assert grpo[0].item() == pytest.approx(1.5, rel=5e-4)
    assert grpo[1].item() == pytest.approx(-0.5, rel=5e-4)


def test_constant_group_has_zero_group_relative_signal() -> None:
    for rewards in ([0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]):
        torch.testing.assert_close(grpo_advantages(rewards), torch.zeros(4))
        torch.testing.assert_close(dr_grpo_advantages(rewards), torch.zeros(4))
        torch.testing.assert_close(rloo_advantages(rewards), torch.zeros(4))


def test_reinforce_zero_reward_has_zero_gradient() -> None:
    log_probs = torch.tensor([-2.0, -3.0], requires_grad=True)
    loss = reinforce_sequence_loss(log_probs, [0.0, 0.0])
    loss.backward()
    torch.testing.assert_close(log_probs.grad, torch.zeros_like(log_probs))


def test_ppo_clip_reference_clips_large_positive_ratio() -> None:
    log_ratio = torch.log(torch.tensor([2.0]))
    advantages = torch.tensor([1.0])
    loss = ppo_clipped_surrogate_loss(log_ratio, advantages, epsilon=0.2)
    assert loss.item() == pytest.approx(-1.2)


def test_mixed_group_probability_matches_binary_dead_zone_formula() -> None:
    assert mixed_group_probability(0.0, 4) == pytest.approx(0.0)
    assert mixed_group_probability(1.0, 4) == pytest.approx(0.0)
    assert mixed_group_probability(0.5, 4) == pytest.approx(0.875)
