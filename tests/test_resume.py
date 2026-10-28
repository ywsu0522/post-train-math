import copy
import json

import pytest

from post_train_math import rloo
from post_train_math.checkpoints import load_checkpoint, validate_checkpoint
from post_train_math.models import MODELS
from post_train_math.rl_core import train_loop
from post_train_math.training import get_method


def assert_same_state(expected, actual):
    import torch

    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert expected.keys() == actual.keys()
        for key in expected:
            assert_same_state(expected[key], actual[key])
    elif isinstance(expected, (list, tuple)):
        assert len(expected) == len(actual)
        for left, right in zip(expected, actual, strict=True):
            assert_same_state(left, right)
    else:
        assert actual == expected


def test_checkpoint_reloads_for_inference_without_final_report(tmp_path, tiny_training):
    import torch
    from peft import PeftModel

    from post_train_math.evaluation import adapter_metadata

    args = tiny_training(tmp_path / "trained", max_steps=2)
    train_loop(*args, method=rloo)
    model = args[0]
    model.eval()
    inputs = {"input_ids": torch.tensor([[3, 5]]), "attention_mask": torch.ones((1, 2))}
    with torch.no_grad():
        expected = model(**inputs).logits.clone()
    checkpoint = args[4].output_dir / "checkpoint-2"
    assert not (checkpoint.parent / "train_report.json").exists()
    metadata = adapter_metadata(checkpoint, MODELS["olmo2"], "data-hash")
    assert metadata["rollout_iterations"] == metadata["optimizer_steps"] == 2
    fresh = tiny_training(tmp_path / "fresh")[0]
    reloaded = PeftModel.from_pretrained(
        fresh.unload(), checkpoint, local_files_only=True
    )
    reloaded.eval()
    with torch.no_grad():
        torch.testing.assert_close(reloaded(**inputs).logits, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "method_name,stop_step",
    [("rloo", 1), ("rloo", 3), ("rloo", 21), ("gr_reinforce", 3)],
)
def test_checkpoint_restores_adamw_rng_and_sampling_across_warmup(
    tmp_path, tiny_training, stop_step, method_name
):
    method = get_method(method_name)
    continuous = tiny_training(
        tmp_path / "continuous", max_steps=23, method=method_name
    )
    expected_history = train_loop(*continuous, method=method)
    expected_weights = {
        n: p.detach().clone() for n, p in continuous[0].named_parameters()
    }
    expected_optimizer = copy.deepcopy(continuous[3].state_dict())
    partial = tiny_training(
        tmp_path / "resumed", max_steps=stop_step, method=method_name
    )
    train_loop(*partial, method=method)
    checkpoint = partial[4].output_dir / f"checkpoint-{stop_step}"
    resumed = tiny_training(tmp_path / "resumed", max_steps=23, method=method_name)
    history = train_loop(*resumed, method=method, resume=checkpoint)
    assert history == expected_history
    assert history[0]["learning_rate"] == pytest.approx(1e-3 / 20)
    assert history[19]["learning_rate"] == history[-1]["learning_rate"] == 1e-3
    assert_same_state(expected_weights, dict(resumed[0].named_parameters()))
    assert_same_state(expected_optimizer, resumed[3].state_dict())
    state = json.loads(
        (resumed[4].output_dir / "checkpoint-23/trainer_state.json").read_text()
    )
    assert state["global_step"] == 23
    assert state["rollout_iterations"] == state["optimizer_steps"] == 23
    assert history[-1]["rollout_iterations"] == 23
    assert history[-1]["optimizer_steps"] == 23


def test_legacy_single_counter_checkpoint_resumes(tmp_path, tiny_training):
    partial = tiny_training(tmp_path / "run", max_steps=1)
    train_loop(*partial, method=rloo)
    checkpoint = partial[4].output_dir / "checkpoint-1"
    state_path = checkpoint / "trainer_state.json"
    state = json.loads(state_path.read_text())
    del state["rollout_iterations"], state["optimizer_steps"]
    state_path.write_text(json.dumps(state))
    definition_path = checkpoint / "run_definition.json"
    definition = json.loads(definition_path.read_text())
    definition["engine"] = "explicit-rloo-v1"
    for key in ("method", "objective", "warmup_steps_unit"):
        del definition["training"][key]
    definition_path.write_text(json.dumps(definition))
    resumed = tiny_training(tmp_path / "run", max_steps=2)
    history = train_loop(*resumed, method=rloo, resume=checkpoint)
    assert history[-1]["rollout_iterations"] == 2
    assert history[-1]["optimizer_steps"] == 2


def test_warmup_uses_restored_optimizer_counter_not_rollout_iteration(
    tmp_path, tiny_training, monkeypatch
):
    from post_train_math import rl_core

    args = tiny_training(tmp_path, max_steps=2)
    monkeypatch.setattr(rl_core, "load_checkpoint", lambda *_: (1, 7, []))
    history = train_loop(*args, method=rloo, resume=tmp_path / "restored")
    assert history[-1]["rollout_iterations"] == 2
    assert history[-1]["optimizer_steps"] == 8
    assert history[-1]["learning_rate"] == pytest.approx(args[4].learning_rate * 8 / 20)


def test_incomplete_incompatible_and_inference_only_checkpoints_are_rejected(
    tmp_path, tiny_training
):
    args = tiny_training(tmp_path, max_steps=1)
    train_loop(*args, method=rloo)
    checkpoint = tmp_path / "checkpoint-1"
    definition = args[-1]
    validate_checkpoint(checkpoint, definition, 2)
    with pytest.raises(ValueError, match="max-steps must exceed"):
        validate_checkpoint(checkpoint, definition, 1)
    changed = copy.deepcopy(definition)
    changed["training"]["learning_rate"] *= 2
    with pytest.raises(ValueError, match="settings differ"):
        validate_checkpoint(checkpoint, changed, 2)
    with pytest.raises(ValueError, match="optimizer.pt"):
        validate_checkpoint(tmp_path / "adapter", definition, 2)
    (checkpoint / "optimizer.pt").unlink()
    with pytest.raises(ValueError, match="optimizer.pt"):
        load_checkpoint(checkpoint, args[0], args[3], args[5], definition, 2)


@pytest.mark.parametrize("method_name", ["rloo", "gr_reinforce"])
def test_cli_training_setup_runs_real_loop_and_exports_evaluable_adapter(
    tmp_path, tiny_training, monkeypatch, method_name
):
    import pandas as pd
    import torch
    import transformers

    from post_train_math import training
    from post_train_math.cli import build_parser
    from post_train_math.evaluation import adapter_metadata

    tiny = tiny_training(tmp_path / "template")
    base = tiny[0].unload()
    tokenizer = tiny[1]
    frame = pd.DataFrame(
        [
            {
                "problem": text,
                "type": "Algebra",
                "level": "Level 1",
                "gt_numerator": 1,
                "gt_denominator": 1,
            }
            for text in ("one", "two")
        ]
    )
    monkeypatch.setattr(training, "FROZEN_TRAIN_ROWS", 2)
    monkeypatch.setattr(
        training,
        "validate_processed_manifest",
        lambda _: ({"cohort": {"name": "boxed-numeric"}}, "data-hash"),
    )
    monkeypatch.setattr(training, "load_cohort", lambda *_: frame)
    monkeypatch.setattr(training, "validate_model_source", lambda *_: {"pinned": True})
    monkeypatch.setattr(training, "load_tokenizer", lambda _: tokenizer)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        lambda *_args, **_kwargs: base,
    )
    build_config = training.build_config

    def cpu_config(*args, **kwargs):
        config = build_config(*args, **kwargs, use_cpu=True)
        config.max_completion_length = 2
        config.save_steps = 1
        return config

    monkeypatch.setattr(training, "build_config", cpu_config)
    args = build_parser().parse_args(
        [
            "train",
            "--method",
            method_name,
            "--model-key",
            "olmo2",
            "--output-dir",
            str(tmp_path / "run"),
            "--prompts-per-update",
            "2",
            "--rollouts",
            "2",
            "--per-device-train-batch-size",
            "2",
            "--gradient-accumulation-steps",
            "2",
            "--max-steps",
            "2",
        ]
    )
    training.run_train(args)
    report = json.loads((args.output_dir / "train_report.json").read_text())
    assert report["training_config"]["method"] == method_name
    assert report["training_config"]["beta"] == get_method(method_name).DEFAULT_BETA
    assert report["global_step"] == len(report["log_history"]) == 2
    assert report["rollout_iterations"] == report["optimizer_steps"] == 2
    assert (args.output_dir / "training_subset.jsonl").is_file()
    metadata = adapter_metadata(
        args.output_dir / "adapter", MODELS["olmo2"], "data-hash"
    )
    assert metadata["rollout_iterations"] == metadata["optimizer_steps"] == 2
    assert metadata["prompts_per_rollout_iteration"] == 2


def test_resume_rejects_a_different_rl_method(tmp_path, tiny_training):
    original = tiny_training(tmp_path / "rloo", max_steps=1)
    train_loop(*original, method=rloo)
    other = tiny_training(tmp_path / "gr", method="gr_reinforce")
    # Even the same numeric beta does not make the objectives interchangeable.
    other[-1]["training"]["beta"] = original[-1]["training"]["beta"]
    with pytest.raises(ValueError, match="settings differ"):
        validate_checkpoint(tmp_path / "rloo/checkpoint-1", other[-1], 2)
