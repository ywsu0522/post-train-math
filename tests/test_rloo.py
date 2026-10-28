import json
from pathlib import Path

import pytest

from post_train_math import rloo
from post_train_math.answers import numeric_reward
from post_train_math.training import get_method, validate_batch
from post_train_math.training_data import prepare_training_subset


def fixed_rollouts(records):
    from post_train_math.rl_core import Rollouts

    return Rollouts(
        [records[0], records[0], records[1], records[1]],
        [[3], [3], [4, 3], [4, 3]],
        [[6, 1], [7, 1], [6, 6, 1], [7, 7, 1]],
        [r"\boxed{1}", r"\boxed{2}", r"\boxed{1}", r"\boxed{2}"],
        [0, 1],
    )


@pytest.mark.parametrize("model_key", ["olmo2", "granite"])
@pytest.mark.parametrize("method_name", ["rloo", "gr_reinforce"])
def test_three_stages_verify_reward_gradients_and_real_lora_updates(
    tmp_path, tiny_training, monkeypatch, model_key, method_name
):
    import torch

    from post_train_math.rl_core import train_loop
    from post_train_math.training import LORA_TARGET_MODULES

    args = tiny_training(
        tmp_path / model_key, max_steps=2, model_key=model_key, method=method_name
    )
    model, _tokenizer, _records, _optimizer, config, accelerator, _definition = args
    frozen = {
        n: p.detach().clone()
        for n, p in model.named_parameters()
        if not p.requires_grad
    }

    def generate(**inputs):
        # Known correct/incorrect answers exercise the actual numeric verifier.
        answers = torch.tensor([[6, 1], [7, 1]], device=inputs["input_ids"].device)
        return torch.cat([inputs["input_ids"], answers], dim=1)

    monkeypatch.setattr(accelerator.unwrap_model(model), "generate", generate)
    history = train_loop(*args, method=get_method(method_name))
    assert len(history) == 2 and all(row["updated"] for row in history)
    for row in history:
        assert row["changed_lora_tensors"] > 0
        assert row["lora_parameter_delta_norm"] > 0
        assert row["gradient_norm_after_clipping"] <= config.max_grad_norm + 1e-6
        assert row["base_gradient_tensors"] == 0
        assert row["lora_gradient_tensors"] == 14
        assert set(row["lora_gradient_norm_by_target"]) == set(LORA_TARGET_MODULES)
        assert row["policy_log_probs_require_grad"]
        assert not row["reference_log_probs_require_grad"]
    for name, parameter in model.named_parameters():
        if name in frozen:
            torch.testing.assert_close(parameter, frozen[name], rtol=0, atol=0)
    events = [
        json.loads(line)
        for line in (config.output_dir / "iteration_log.jsonl").read_text().splitlines()
    ]
    for iteration in (1, 2):
        assert [
            event["stage"]
            for event in events
            if event["iteration"] == iteration and event.get("status") == "complete"
        ] == ["1.1", "1.2", "2"]
        advantages = next(
            event
            for event in events
            if event["iteration"] == iteration and event.get("status") == "advantages"
        )
        assert advantages["verifier_rewards"] == [1, 0, 1, 0]
    assert events[2]["completions"] == config.generation_batch_size


def test_microbatch_estimator_matches_direct_full_batch_autograd(
    tmp_path, tiny_training
):
    import torch

    from post_train_math.rl_core import (
        GradientContext,
        sequence_log_probs,
        verify_lora_targets,
    )

    model, tok, records, opt, cfg, acc, _ = tiny_training(tmp_path, microbatch=1)
    rollouts = fixed_rollouts(records)
    trainable = verify_lora_targets(model)
    # Nonzero adapters ensure reference and current policy differ for the KL reward.
    with torch.no_grad():
        for name, p in trainable.items():
            if ".lora_B." in name:
                p.normal_(std=0.05)
    model.eval()
    with torch.no_grad(), acc.unwrap_model(model).disable_adapter():
        ref = sequence_log_probs(model, tok, rollouts, 0, 4, acc.device)
    pi = sequence_log_probs(model, tok, rollouts, 0, 4, acc.device)
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0]) - cfg.beta * (pi.detach() - ref)
    # For G=2, each sibling is exactly the other's leave-one-out baseline.
    expected_advantage = (
        rewards.reshape(2, 2) - rewards.reshape(2, 2).flip(1)
    ).flatten()
    (-(expected_advantage * pi).mean()).backward()
    expected = {n: p.grad.clone() for n, p in trainable.items()}
    stats = rloo.estimate_gradients(
        GradientContext(model, tok, opt, cfg, acc, 1, trainable), rollouts
    )
    assert any(p.norm() > 0 for p in expected.values())
    for name, p in trainable.items():
        torch.testing.assert_close(p.grad, expected[name], rtol=2e-4, atol=2e-6)
    assert not stats["reference_log_probs_require_grad"]


def test_log_prob_sums_exclude_prompts_and_padding_but_include_eos(
    tmp_path, tiny_training
):
    import torch

    from post_train_math.rl_core import sequence_log_probs

    model, tok, records, _, _, acc, _ = tiny_training(tmp_path)
    rollouts = fixed_rollouts(records)
    model.eval()
    actual = sequence_log_probs(model, tok, rollouts, 0, 4, acc.device)
    for i, (prompt, answer) in enumerate(
        zip(rollouts.prompt_ids, rollouts.completion_ids, strict=True)
    ):
        ids = torch.tensor([prompt + answer])
        logits = model(input_ids=ids).logits[0, len(prompt) - 1 : -1].float()
        expected = logits.log_softmax(-1).gather(1, torch.tensor(answer)[:, None]).sum()
        torch.testing.assert_close(actual[i], expected, rtol=1e-6, atol=1e-6)


def test_sampling_is_with_replacement_and_groups_remain_aligned(
    tmp_path, tiny_training
):
    from post_train_math.rl_core import generate_rollouts

    model, tok, records, _, cfg, acc, _ = tiny_training(tmp_path)
    cfg.prompts_per_update = 8  # More draws than rows guarantees repeated prompts.
    cfg.gradient_accumulation_steps = 8
    batch = generate_rollouts(model, tok, records, cfg, acc, 1)
    assert len(batch.sampled_indices) == 8
    assert len(set(batch.sampled_indices)) < 8
    assert len(batch.completion_ids) == len(batch.texts) == 16
    assert batch.records == [
        records[i] for i in batch.sampled_indices for _ in range(2)
    ]
    repeated = generate_rollouts(model, tok, records, cfg, acc, 1)
    assert repeated.sampled_indices == batch.sampled_indices


def test_loo_is_group_local_and_zero_signal_is_reported_honestly(
    tmp_path, tiny_training
):
    import torch

    from post_train_math.rl_core import train_loop
    from post_train_math.rloo import leave_one_out_advantages

    rewards = torch.tensor([1.0, 0.0, 0.0, 0.0, 7.0, 7.0, 7.0, 7.0], requires_grad=True)
    advantages = leave_one_out_advantages(rewards, 4)
    torch.testing.assert_close(
        advantages, torch.tensor([1.0, -1 / 3, -1 / 3, -1 / 3, 0, 0, 0, 0])
    )
    assert not advantages.requires_grad
    args = tiny_training(tmp_path, max_steps=1)
    args[4].beta = 0
    history = train_loop(
        *args, method=rloo, reward_fn=lambda completions, **_: [0.0] * len(completions)
    )
    assert not history[0]["updated"]
    assert history[0]["lora_parameter_delta_norm"] == 0


def test_warmup_then_constant_lr_is_used_by_actual_adamw_steps(tmp_path, tiny_training):
    import torch

    from post_train_math.rl_core import update_parameters, verify_lora_targets

    model, _, _, opt, cfg, acc, _ = tiny_training(tmp_path)
    trainable = verify_lora_targets(model)
    rates = []
    for iteration in range(1, 26):
        for p in trainable.values():
            p.grad = torch.ones_like(p)
        result = update_parameters(opt, cfg, acc, iteration, trainable)
        assert result["updated"]
        rates.append(result["learning_rate"])
    assert rates[0] == pytest.approx(cfg.learning_rate / 20)
    assert rates[19:] == [cfg.learning_rate] * 6
    assert all(state["step"] == 25 for state in opt.state_dict()["state"].values())


def test_lora_audit_rejects_unfrozen_output_head(tmp_path, tiny_training):
    from post_train_math.rl_core import verify_lora_targets

    model = tiny_training(tmp_path)[0]
    model.get_output_embeddings().weight.requires_grad_(True)
    with pytest.raises(ValueError, match="Non-LoRA trainable parameter"):
        verify_lora_targets(model)


def test_amp_skipped_step_aborts_instead_of_reporting_success():
    from types import SimpleNamespace
    from unittest.mock import Mock

    import torch

    from post_train_math.rl_core import update_parameters

    parameter = torch.nn.Parameter(torch.ones(1))
    parameter.grad = torch.ones_like(parameter)
    optimizer = SimpleNamespace(param_groups=[{}], step=Mock(), zero_grad=Mock())
    config = SimpleNamespace(learning_rate=1e-5, warmup_steps=20, max_grad_norm=0.5)
    accelerator = SimpleNamespace(optimizer_step_was_skipped=True)
    with pytest.raises(RuntimeError, match="aborted on overflow"):
        update_parameters(optimizer, config, accelerator, 1, {"lora": parameter})
    optimizer.step.assert_called_once()
    optimizer.zero_grad.assert_not_called()


def test_reward_supports_trl_text_and_conversations():
    assert numeric_reward(
        [
            r"\boxed{2/4}",
            [{"role": "assistant", "content": r"\boxed{50\%}"}],
            r"\boxed{1/2} then \boxed{",
            r"\boxed{3}",
        ],
        [1, 1, 1, 1],
        [2, 2, 2, 2],
    ) == [1.0, 1.0, 0.0, 0.0]
    with pytest.raises(ValueError):
        numeric_reward([r"\boxed{1}"], [], [])


@pytest.mark.parametrize("world_size,expected", [(1, 32), (2, 16), (4, 8)])
def test_b_counts_prompts_and_r_counts_rollouts(world_size, expected):
    validate_batch(8, 4, 1, world_size, expected)


@pytest.mark.parametrize(
    "batch",
    [
        (0, 4, 1, 1, 32),
        (8, 1, 1, 1, 32),
        (3, 2, 4, 1, 2),
        (8, 4, 1, 1, 0),
        (8, 4, 1, 1, 16),
    ],
)
def test_invalid_batches_are_rejected(batch):
    with pytest.raises(ValueError):
        validate_batch(*batch)


def test_native_template_length_filter_and_saved_definition(tmp_path):
    import hashlib

    import pandas as pd

    from post_train_math.models import MODELS

    frame = pd.DataFrame(
        [
            {
                "problem": text,
                "type": "Geometry",
                "level": f"Level {index}",
                "gt_numerator": index,
                "gt_denominator": 1,
            }
            for index, text in enumerate(("short", "at boundary", "too long"), 1)
        ]
    )
    original = frame.copy(deep=True)

    class Tokenizer:
        def __init__(self, extra=0):
            self.extra = extra
            self.special_tokens_map = {"bos_token": "BOS", "eos_token": "EOS"}

        def get_chat_template(self):
            return f"native-template-{self.extra}"

        def get_vocab(self):
            return {"BOS": 0, "EOS": 1}

        def apply_chat_template(self, messages, **kwargs):
            assert kwargs == {
                "tokenize": True,
                "add_generation_prompt": True,
                "truncation": False,
                "padding": False,
                "return_dict": True,
            }
            assert messages[0]["role"] == "system"
            problem = messages[1]["content"].removeprefix("Problem:\n")
            return {
                "input_ids": [0]
                * (
                    {"short": 100, "at boundary": 2048, "too long": 2049}[problem]
                    + self.extra
                )
            }

    records, manifest = prepare_training_subset(
        frame,
        Tokenizer(),
        MODELS["olmo2"],
        "fixed-data-hash",
        tmp_path / "first",
    )
    assert [row["prompt_tokens"] for row in records] == [100, 2048]
    assert manifest["counts"] == {"cohort_train": 3, "selected": 2, "overlong": 1}
    assert manifest["data_manifest_sha256"] == "fixed-data-hash"
    assert manifest["selection"] == {
        "field": "prompt_tokens",
        "operator": "<=",
        "threshold": 2048,
    }
    excluded = [
        json.loads(row)
        for row in (tmp_path / "first/overlong_prompts.jsonl").read_text().splitlines()
    ]
    assert (
        excluded[0]["problem_id"] == "math-" + hashlib.sha256(b"too long").hexdigest()
    )
    assert excluded[0]["prompt_tokens"] == 2049
    assert excluded[0]["type"] == "Geometry" and excluded[0]["level"] == "Level 3"
    assert not excluded[0]["included"]
    audit = [
        json.loads(row)
        for row in (tmp_path / "first/prompt_audit.jsonl").read_text().splitlines()
    ]
    assert len(audit) == 3 and sum(row["included"] for row in audit) == 2
    for name, digest in manifest["files_sha256"].items():
        assert (
            hashlib.sha256((tmp_path / "first" / name).read_bytes()).hexdigest()
            == digest
        )
    assert [
        json.loads(row)
        for row in (tmp_path / "first/training_subset.jsonl").read_text().splitlines()
    ] == records
    _, repeated = prepare_training_subset(
        frame, Tokenizer(), MODELS["olmo2"], "fixed-data-hash", tmp_path / "repeat"
    )
    assert manifest == repeated
    other_records, other = prepare_training_subset(
        frame,
        Tokenizer(extra=1),
        MODELS["granite"],
        "fixed-data-hash",
        tmp_path / "other",
    )
    assert len(other_records) == 1 and other["counts"]["overlong"] == 2
    assert other_records[0]["problem_id"] == records[0]["problem_id"]
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("model_key", ["olmo2", "granite"])
def test_saved_native_tokenizer_audit_matches_frozen_contract(model_key):
    import hashlib

    from post_train_math.models import MODELS

    root = Path(__file__).resolve().parents[1]
    folder = root / "cohort/training-subsets" / model_key
    definition = json.loads(
        (folder / "training_subset_manifest.json").read_text(encoding="utf-8")
    )
    assert (
        definition["data_manifest_sha256"]
        == hashlib.sha256((root / "cohort/manifest.json").read_bytes()).hexdigest()
    )
    assert definition["cohort"] == "boxed-numeric"
    spec = MODELS[model_key]
    assert definition["model"] == {
        "key": spec.key,
        "repo_id": spec.repo_id,
        "revision": spec.revision,
    }
    assert definition["counts"]["cohort_train"] == 5589
    assert definition["tokenization"]["truncation"] is False
    assert definition["tokenization"]["add_generation_prompt"] is True
    for name, digest in definition["files_sha256"].items():
        assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == digest
    audit = [
        json.loads(line)
        for line in (folder / "prompt_audit.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    selected = [
        json.loads(line)
        for line in (folder / "training_subset.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(audit) == len({record["problem_id"] for record in audit}) == 5589
    assert {record["problem_id"] for record in selected} == {
        record["problem_id"] for record in audit if record["prompt_tokens"] <= 2048
    }
    assert len(selected) == definition["counts"]["selected"]
