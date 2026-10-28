"""Mathematical and microbatch/rank invariants for GR-REINFORCE."""

import math
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from post_train_math import gr_reinforce
from post_train_math.rl_core import (
    CompletionLogProbs,
    GradientContext,
    Rollouts,
    completion_log_probs,
    verify_lora_targets,
)


def fixed_rollouts(records):
    return Rollouts(
        [records[0]] * 4,
        [[3]] * 4,
        [[6, 1], [7, 7, 1], [6, 6, 6, 1], [7, 7, 7, 7, 1]],
        [r"\boxed{1}", r"\boxed{2}", r"\boxed{1}", r"\boxed{2}"],
        [0],
    )


def nonzero_adapters(model):
    import torch

    with torch.no_grad():
        for name, p in verify_lora_targets(model).items():
            if ".lora_B." in name:
                p.normal_(std=0.2)


def direct_gradients(args, rollouts):
    """Independent full-rollout expression, with all valid KL tokens concatenated."""
    import torch

    model, tok, _, opt, cfg, acc, _ = args
    opt.zero_grad(set_to_none=True)
    model.eval()
    with torch.no_grad(), acc.unwrap_model(model).disable_adapter():
        ref = completion_log_probs(model, tok, rollouts, 0, 4, acc.device)
    pi = completion_log_probs(model, tok, rollouts, 0, 4, acc.device)
    # Two successes/two failures: population std of rewards is 0.5.
    adv = torch.tensor([0.5, -0.5, 0.5, -0.5]) / (0.5 + 1e-6)
    seq_means = torch.stack(
        [
            pi.values[i, : len(ids)].mean()
            for i, ids in enumerate(rollouts.completion_ids)
        ]
    )
    deltas = torch.cat(
        [
            ref.values[i, : len(ids)] - pi.values[i, : len(ids)]
            for i, ids in enumerate(rollouts.completion_ids)
        ]
    ).clamp(-20, 20)
    loss = -(adv * seq_means).mean() + cfg.beta * (deltas.exp() - deltas - 1).mean()
    loss.backward()
    return loss.item(), {
        name: p.grad.detach().clone() for name, p in verify_lora_targets(model).items()
    }


def test_population_std_epsilon_clipping_and_zero_variance():
    import torch

    rewards = torch.tensor([1.0, 0.0, 0.0, 0.0, 7.0, 7.0, 7.0, 7.0], requires_grad=True)
    actual = gr_reinforce.group_relative_advantages(rewards, 4)
    expected = torch.tensor([0.75, -0.25, -0.25, -0.25]) / (math.sqrt(3) / 4 + 1e-6)
    torch.testing.assert_close(actual[:4], expected)
    assert torch.equal(actual[4:], torch.zeros(4))
    assert not actual.requires_grad
    outlier = torch.zeros(100)
    outlier[0] = 1
    assert gr_reinforce.group_relative_advantages(outlier, 100)[0] == 5
    for size in (1, 3):
        with pytest.raises(ValueError, match="complete groups"):
            gr_reinforce.group_relative_advantages(rewards, size)


def test_loss_gradients_match_closed_form_and_detach_fixed_inputs():
    import torch

    values = torch.tensor([[-1.0, -2.0, 9.0], [-3.0, -4.0, -5.0]], requires_grad=True)
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    delta = torch.tensor([[0.2, -0.1, 100.0], [-0.5, 0.3, 0.1]])
    reference = (values.detach() + delta).requires_grad_()
    advantages = torch.tensor([1.0, -1.0], requires_grad=True)
    terms = gr_reinforce.loss_terms(
        CompletionLogProbs(values, mask),
        reference,
        advantages,
        sequence_count=2,
        token_weight=1 / 5,
        beta=0.05,
    )
    terms.loss.backward()
    expected = mask * (
        -advantages.detach()[:, None] / (2 * mask.sum(1)[:, None])
        + 0.05 * (1 - delta.clamp(-20, 20).exp()) / 5
    )
    torch.testing.assert_close(values.grad, expected, rtol=1e-5, atol=1e-7)
    assert reference.grad is None and advantages.grad is None
    assert values.grad[0, 2] == 0
    # A zero policy-gradient signal leaves the independent KL gradient active.
    assert terms.metrics["kl_loss"].item() > 0


@pytest.mark.parametrize("microbatch", [1, 2, 4])
def test_estimator_matches_full_rollout_autograd_for_unequal_lengths(
    tmp_path, tiny_training, microbatch
):
    import torch

    args = tiny_training(tmp_path, microbatch=microbatch, method="gr_reinforce")
    model, tok, records, opt, cfg, acc, _ = args
    cfg.num_generations = 4
    nonzero_adapters(model)
    rollouts = fixed_rollouts(records)
    expected_loss, expected = direct_gradients(args, rollouts)
    trainable = verify_lora_targets(model)
    stats = gr_reinforce.estimate_gradients(
        GradientContext(model, tok, opt, cfg, acc, 1, trainable), rollouts
    )
    assert stats["loss"] == pytest.approx(expected_loss, rel=2e-4, abs=2e-6)
    for name, parameter in trainable.items():
        torch.testing.assert_close(parameter.grad, expected[name], rtol=3e-4, atol=2e-6)
    assert not stats["reference_log_probs_require_grad"]
    assert stats["policy_log_probs_require_grad"]
    assert stats["approx_kl"] > 0
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)


def test_rank_average_matches_global_gradient_with_group_spanning_ranks(
    tmp_path, tiny_training
):
    import torch

    full = tiny_training(tmp_path / "full", method="gr_reinforce")
    full[4].num_generations = 4
    nonzero_adapters(full[0])
    rollouts = fixed_rollouts(full[2])
    _, expected = direct_gradients(full, rollouts)
    rank_gradients = []
    for rank in (0, 1):
        args = tiny_training(
            tmp_path / f"rank-{rank}", microbatch=1, method="gr_reinforce"
        )
        model, tok, records, opt, cfg, acc, _ = args
        cfg.num_generations = 4
        nonzero_adapters(model)
        start, end = rank * 2, (rank + 1) * 2
        batch = fixed_rollouts(records)
        local = Rollouts(
            batch.records[start:end],
            batch.prompt_ids[start:end],
            batch.completion_ids[start:end],
            batch.texts[start:end],
            [0],
        )
        gathers = 0

        def gather(tensor):
            nonlocal gathers
            gathers += 1
            # Reward gather reassembles the single group across both ranks.
            return torch.tensor([1.0, 0.0, 1.0, 0.0]) if gathers == 1 else tensor

        distributed = SimpleNamespace(
            is_main_process=rank == 0,
            process_index=rank,
            num_processes=2,
            device=acc.device,
            unwrap_model=acc.unwrap_model,
            autocast=acc.autocast,
            no_sync=lambda _: nullcontext(),
            backward=acc.backward,
            unscale_gradients=acc.unscale_gradients,
            gather=gather,
            reduce=lambda tensor, reduction: (
                torch.tensor(14.0) if reduction == "sum" else tensor
            ),
        )
        trainable = verify_lora_targets(model)
        gr_reinforce.estimate_gradients(
            GradientContext(model, tok, opt, cfg, distributed, 1, trainable), local
        )
        rank_gradients.append({n: p.grad.clone() for n, p in trainable.items()})
    for name, gradient in expected.items():
        average = (rank_gradients[0][name] + rank_gradients[1][name]) / 2
        torch.testing.assert_close(average, gradient, rtol=3e-4, atol=2e-6)


def test_zero_reward_variance_retains_kl_gradient(tmp_path, tiny_training):
    args = tiny_training(tmp_path, method="gr_reinforce")
    model, tok, records, opt, cfg, acc, _ = args
    nonzero_adapters(model)
    trainable = verify_lora_targets(model)
    stats = gr_reinforce.estimate_gradients(
        GradientContext(model, tok, opt, cfg, acc, 1, trainable),
        fixed_rollouts(records),
        reward_fn=lambda completions, **_: [1.0] * len(completions),
    )
    assert stats["policy_loss"] == 0
    assert stats["kl_loss"] > 0
    assert any(p.grad.norm() > 0 for p in trainable.values())
