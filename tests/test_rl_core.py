"""The method boundary must expose ready gradients before the core updates."""

from types import SimpleNamespace

import pytest

from post_train_math import rl_core, rloo
from post_train_math.training import get_method


def test_core_updates_only_after_method_populates_gradients(
    tmp_path, tiny_training, monkeypatch
):
    args = tiny_training(tmp_path, max_steps=1)
    calls = []
    original_update = rl_core.update_parameters

    def estimate(context, rollouts, **kwargs):
        assert not context.optimizer.state_dict()["state"]
        stats = rloo.estimate_gradients(context, rollouts, **kwargs)
        assert all(p.grad is not None for p in context.trainable.values())
        assert not context.optimizer.state_dict()["state"]
        calls.append("gradients_ready")
        return stats

    def update(*positional, **kwargs):
        assert calls == ["gradients_ready"]
        result = original_update(*positional, **kwargs)
        calls.append("parameters_updated")
        return result

    monkeypatch.setattr(rl_core, "update_parameters", update)
    method = SimpleNamespace(NAME="rloo", estimate_gradients=estimate)
    rl_core.train_loop(*args, method=method)
    assert calls == ["gradients_ready", "parameters_updated"]
    assert args[3].state_dict()["state"]


def test_core_rejects_method_returning_without_gradients(tmp_path, tiny_training):
    args = tiny_training(tmp_path, max_steps=1)
    method = SimpleNamespace(NAME="rloo", estimate_gradients=lambda *a, **k: {})
    with pytest.raises(RuntimeError, match="without ready LoRA gradients"):
        rl_core.train_loop(*args, method=method)
    assert not args[3].state_dict()["state"]
    assert not (tmp_path / "checkpoint-1").exists()


def test_method_selection_and_defaults_are_explicit(tmp_path):
    from post_train_math.cli import build_parser
    from post_train_math.training import TrainConfig

    config = TrainConfig(tmp_path, method="gr_reinforce")
    assert config.beta == 0.05
    assert config.to_dict()["objective"]["kl_reduction"] == "global_rollout_token_mean"
    assert TrainConfig(tmp_path).beta == 0.01
    with pytest.raises(ValueError, match="Unknown RL method"):
        get_method("missing")
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "train",
                "--model-key",
                "olmo2",
                "--output-dir",
                str(tmp_path),
                "--method",
                "missing",
            ]
        )
