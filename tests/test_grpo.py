"""Clipped gradients, rollout reuse and cross-rank normalization for GRPO."""

import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from test_gr_reinforce import fixed_rollouts, nonzero_adapters

from post_train_math import gr_reinforce, grpo, rl_core
from post_train_math.rl_core import CompletionLogProbs, GradientContext


def test_clipped_surrogate_gradients_masks_and_detached_inputs():
    import torch

    values = torch.full((2, 5), -3.0, requires_grad=True)
    log_ratio = torch.tensor([[0.5, -0.5, 0.1, 25.0, 80.0]] * 2)
    old = (values.detach() - log_ratio).requires_grad_()
    reference = (values.detach() + 0.1).requires_grad_()
    advantages = torch.tensor([1.0, -1.0], requires_grad=True)
    mask = torch.tensor([[1.0, 1.0, 1.0, 1.0, 0.0]] * 2)
    terms = grpo.loss_terms(
        CompletionLogProbs(values, mask),
        old,
        reference,
        advantages,
        sequence_count=2,
        token_weight=1 / 8,
        beta=0.05,
    )
    terms.loss.backward()
    # Positive advantages stop at the upper bound; negative ones at the lower.
    # Values outside the log-ratio clamp also have zero policy gradient.
    ratio = log_ratio.clamp(-20, 20).exp()
    active = torch.tensor([[0, 1, 1, 0, 0], [1, 0, 1, 0, 0]])
    expected = -advantages.detach()[:, None] * ratio * active / 8
    expected += mask * 0.05 * (1 - torch.exp(torch.tensor(0.1))) / 8
    torch.testing.assert_close(values.grad, expected, rtol=1e-5, atol=1e-7)
    assert old.grad is None and reference.grad is None and advantages.grad is None
    assert terms.metrics["clipfrac"].item() == pytest.approx(6 / 8)
    assert torch.isfinite(terms.loss)


@pytest.mark.parametrize("microbatch", [1, 2, 4])
def test_first_update_matches_gr_reinforce_gradients(
    tmp_path, tiny_training, microbatch
):
    import torch

    args = tiny_training(tmp_path, method="grpo", microbatch=microbatch)
    model, tok, records, opt, cfg, acc, _ = args
    cfg.num_generations = 4
    nonzero_adapters(model)
    batch = fixed_rollouts(records)
    context = GradientContext(
        model, tok, opt, cfg, acc, 1, rl_core.verify_lora_targets(model)
    )
    gr_reinforce.estimate_gradients(context, batch)
    expected = {name: p.grad.clone() for name, p in context.trainable.items()}
    stats = grpo.estimate_gradients(context, batch)
    for name, p in context.trainable.items():
        torch.testing.assert_close(p.grad, expected[name], rtol=3e-4, atol=2e-6)
    assert stats["ratio_mean"] == pytest.approx(1)
    assert stats["clipfrac"] == 0
    assert not stats["old_log_probs_require_grad"]


def change_policy(model):
    import torch

    with torch.no_grad():
        for name, p in rl_core.verify_lora_targets(model).items():
            if ".lora_B." in name:
                p.mul_(1.8)


def full_batch_gradients(args, batch):
    """Independent expression on a changed policy with fixed pre-update inputs."""
    import torch

    model, tok, _, opt, cfg, acc, _ = args
    model.eval()
    with torch.no_grad():
        old = rl_core.completion_log_probs(model, tok, batch, 0, 4, acc.device)
        with model.disable_adapter():
            ref = rl_core.completion_log_probs(model, tok, batch, 0, 4, acc.device)
    change_policy(model)
    opt.zero_grad(set_to_none=True)
    pi = rl_core.completion_log_probs(model, tok, batch, 0, 4, acc.device)
    adv = torch.tensor([0.5, -0.5, 0.5, -0.5]) / (0.5 + 1e-6)
    objectives, divergences = [], []
    for i, ids in enumerate(batch.completion_ids):
        logp = pi.values[i, : len(ids)]
        ratios = (logp - old.values[i, : len(ids)]).clamp(-20, 20).exp()
        objectives.append(
            torch.minimum(ratios * adv[i], ratios.clamp(0.8, 1.2) * adv[i]).mean()
        )
        delta = (ref.values[i, : len(ids)] - logp).clamp(-20, 20)
        divergences.append(delta.exp() - delta - 1)
    loss = -torch.stack(objectives).mean() + cfg.beta * torch.cat(divergences).mean()
    loss.backward()
    return loss.item(), {
        n: p.grad.clone() for n, p in rl_core.verify_lora_targets(model).items()
    }


@pytest.mark.parametrize("microbatch", [1, 2, 4])
def test_reused_rollout_matches_full_batch_gradient(
    tmp_path, tiny_training, microbatch
):
    import torch

    expected_args = tiny_training(tmp_path / "full", method="grpo")
    nonzero_adapters(expected_args[0])
    expected_loss, expected = full_batch_gradients(
        expected_args, fixed_rollouts(expected_args[2])
    )
    args = tiny_training(tmp_path / "micro", method="grpo", microbatch=microbatch)
    model, tok, records, opt, cfg, acc, _ = args
    cfg.num_generations = 4
    nonzero_adapters(model)
    batch = fixed_rollouts(records)
    context = GradientContext(
        model, tok, opt, cfg, acc, 1, rl_core.verify_lora_targets(model)
    )
    prepared = grpo.prepare_rollouts(context, batch)
    old = [p.clone() for p in prepared.old_log_probs]
    change_policy(model)
    stats = grpo.estimate_gradients(context, batch, prepared=prepared)
    assert stats["loss"] == pytest.approx(expected_loss, rel=3e-4, abs=2e-6)
    for n, p in context.trainable.items():
        torch.testing.assert_close(p.grad, expected[n], rtol=3e-4, atol=2e-6)
    for before, after in zip(old, prepared.old_log_probs, strict=True):
        torch.testing.assert_close(after, before, rtol=0, atol=0)


def test_reused_rollout_rank_average_matches_global_gradient(tmp_path, tiny_training):
    import torch

    full = tiny_training(tmp_path / "full", method="grpo")
    nonzero_adapters(full[0])
    expected_loss, expected = full_batch_gradients(full, fixed_rollouts(full[2]))
    gradients, losses = [], []
    for rank in (0, 1):
        model, tok, records, opt, cfg, acc, _ = tiny_training(
            tmp_path / f"rank-{rank}",
            method="grpo",
            microbatch=1,
        )
        cfg.num_generations = 4
        nonzero_adapters(model)
        batch = fixed_rollouts(records)
        start, end = rank * 2, rank * 2 + 2
        local = rl_core.Rollouts(
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
        context = GradientContext(
            model, tok, opt, cfg, distributed, 1, rl_core.verify_lora_targets(model)
        )
        prepared = grpo.prepare_rollouts(context, local)
        change_policy(model)
        losses.append(
            grpo.estimate_gradients(context, local, prepared=prepared)["loss"]
        )
        gradients.append({n: p.grad.clone() for n, p in context.trainable.items()})
    assert sum(losses) / 2 == pytest.approx(expected_loss, rel=3e-4, abs=2e-6)
    for n, expected_gradient in expected.items():
        torch.testing.assert_close(
            (gradients[0][n] + gradients[1][n]) / 2,
            expected_gradient,
            rtol=3e-4,
            atol=2e-6,
        )


def test_core_reuses_one_preparation_and_records_each_update(
    tmp_path, tiny_training, monkeypatch
):
    import torch

    args = tiny_training(tmp_path, max_steps=2, method="grpo")
    batch = fixed_rollouts(args[2])
    generations, preparations = [], []
    estimates = []
    prepare, estimate = grpo.prepare_rollouts, grpo.estimate_gradients

    def generate(*_):
        generations.append(1)
        return batch

    def prepare_once(*a, **kw):
        result = prepare(*a, **kw)
        preparations.append(result)
        return result

    def estimate_again(context, rollouts, **kwargs):
        prepared = kwargs["prepared"]
        assert prepared is preparations[-1]
        old = [value.clone() for value in prepared.old_log_probs]
        estimates.append(prepared)
        result = estimate(context, rollouts, **kwargs)
        for before, after in zip(old, prepared.old_log_probs, strict=True):
            torch.testing.assert_close(after, before, rtol=0, atol=0)
        return result

    monkeypatch.setattr(rl_core, "generate_rollouts", generate)
    monkeypatch.setattr(grpo, "prepare_rollouts", prepare_once)
    monkeypatch.setattr(grpo, "estimate_gradients", estimate_again)
    history = rl_core.train_loop(*args, method=grpo)
    assert len(generations) == len(preparations) == len(history) == 2
    assert len(estimates) == 4
    assert estimates[0] is estimates[1] and estimates[2] is estimates[3]
    assert history[-1]["optimizer_steps"] == 4
    assert [u["optimizer_steps"] for h in history for u in h["updates"]] == [1, 2, 3, 4]
    assert [u["update_in_rollout"] for u in history[0]["updates"]] == [1, 2]
    assert history[0]["updates"][0]["ratio_mean"] == pytest.approx(1)
    events = [
        json.loads(line)
        for line in (tmp_path / "iteration_log.jsonl").read_text().splitlines()
    ]
    assert sum(e.get("status") == "advantages" for e in events) == 2
    assert (
        sum(e.get("stage") == "2" and e.get("status") == "complete" for e in events)
        == 4
    )


def test_checkpoint_rejects_changed_grpo_update_count(tmp_path, tiny_training):
    from post_train_math.checkpoints import run_definition, validate_checkpoint
    from post_train_math.models import MODELS

    args = tiny_training(tmp_path, method="grpo", max_steps=1)
    rl_core.train_loop(*args, method=grpo)
    args[4].num_iterations = 1
    definition = run_definition(
        MODELS["olmo2"], "data-hash", args[4], args[0].peft_config["default"]
    )
    with pytest.raises(ValueError, match="settings differ"):
        validate_checkpoint(tmp_path / "checkpoint-1", definition, 2)
