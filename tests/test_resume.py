import copy

import pytest

from post_train_math.rloo import (
    build_config,
    checkpoint_callback,
    resumable_trainer_class,
    run_definition,
    validate_checkpoint,
)


@pytest.fixture
def tiny_trainer(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    monkeypatch.setenv("TRL_HUB_TELEMETRY", "0")
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from transformers import (
        GPT2Config,
        GPT2LMHeadModel,
        PreTrainedTokenizerFast,
        set_seed,
    )

    from post_train_math.models import MODELS

    threads = torch.get_num_threads()
    torch.set_num_threads(1)

    def make(output, max_steps):
        set_seed(42)
        backend = Tokenizer(
            WordLevel(
                {
                    "[PAD]": 0,
                    "[EOS]": 1,
                    "[UNK]": 2,
                    "one": 3,
                    "two": 4,
                    "three": 5,
                    "four": 6,
                    "answer": 7,
                },
                unk_token="[UNK]",
            )
        )
        backend.pre_tokenizer = WhitespaceSplit()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend,
            pad_token="[PAD]",
            eos_token="[EOS]",
            unk_token="[UNK]",
            padding_side="left",
        )
        tokenizer.chat_template = (
            "{% for message in messages %}{{ message['content'] }} {% endfor %}answer"
        )
        model = GPT2LMHeadModel(
            GPT2Config(
                vocab_size=8,
                n_layer=1,
                n_head=1,
                n_embd=8,
                n_positions=16,
                eos_token_id=1,
                pad_token_id=0,
                bos_token_id=1,
                use_cache=False,
            )
        )
        config = build_config(
            output,
            prompts=1,
            rollouts=2,
            per_device_train_batch_size=2,
            world_size=1,
            gradient_accumulation_steps=1,
            max_steps=max_steps,
            learning_rate=1e-3,
            use_cpu=True,
        )
        assert config.save_steps == 50
        config.max_completion_length = 2
        config.save_steps = 2  # Exercise multiple real saves in a short CPU run.
        config.disable_tqdm = True
        lora = LoraConfig(
            task_type="CAUSAL_LM",
            r=2,
            lora_alpha=4,
            target_modules="all-linear",
            lora_dropout=0.0,
        )
        definition = run_definition(
            MODELS["olmo2"], "data-hash", {"subset": "fixed"}, config, lora
        )

        def reward(completions, **kwargs):
            return [float(i) for i in range(len(completions))]

        trainer = resumable_trainer_class()(
            model=model,
            args=config,
            processing_class=tokenizer,
            peft_config=lora,
            reward_funcs=reward,
            train_dataset=Dataset.from_list(
                [
                    {"prompt": [{"role": "user", "content": prompt}]}
                    for prompt in ("one", "two", "three", "four")
                ]
            ),
            callbacks=[checkpoint_callback(definition)],
        )
        return trainer, definition

    yield make
    torch.set_num_threads(threads)


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


@pytest.mark.parametrize("stop_step", [2, 4])
def test_full_checkpoint_restores_adamw_and_continues_identically(
    tmp_path, tiny_trainer, stop_step
):
    import torch
    from transformers import TrainerCallback

    continuous, _ = tiny_trainer(tmp_path / "continuous", 6)
    continuous.train()
    uninterrupted_weights = {
        key: value.detach().clone()
        for key, value in continuous.model.state_dict().items()
    }
    uninterrupted_optimizer = copy.deepcopy(continuous.optimizer.state_dict())
    partial, definition = tiny_trainer(tmp_path / "resumed", stop_step)
    partial.train()
    checkpoint = tmp_path / f"resumed/checkpoint-{stop_step}"
    validate_checkpoint(checkpoint, definition, max_steps=6)
    saved_optimizer = torch.load(checkpoint / "optimizer.pt", weights_only=True)
    assert saved_optimizer["state"]
    assert all(
        "exp_avg" in state and "exp_avg_sq" in state
        for state in saved_optimizer["state"].values()
    )
    assert any(
        state["exp_avg"].count_nonzero() for state in saved_optimizer["state"].values()
    )
    saved_scheduler = torch.load(checkpoint / "scheduler.pt", weights_only=True)
    for step in range(2, stop_step + 1, 2):
        assert (tmp_path / f"resumed/checkpoint-{step}/run_definition.json").is_file()
    resumed, current_definition = tiny_trainer(tmp_path / "resumed", 6)
    validate_checkpoint(checkpoint, current_definition, max_steps=6)
    restored = []

    class CheckRestored(TrainerCallback):
        def on_train_begin(
            self, args, state, control, optimizer=None, lr_scheduler=None, **kwargs
        ):
            assert state.global_step == stop_step
            assert_same_state(saved_optimizer, optimizer.state_dict())
            assert_same_state(saved_scheduler, lr_scheduler.state_dict())
            restored.append(True)

    resumed.add_callback(CheckRestored())
    result = resumed.train(resume_from_checkpoint=str(checkpoint))
    assert restored and resumed.state.global_step == 6
    assert "train_loss" in result.metrics
    assert {record["step"] for record in resumed.state.log_history} >= {
        1,
        2,
        3,
        4,
        5,
        6,
    }
    assert all(
        record["learning_rate"] == 1e-3
        for record in resumed.state.log_history
        if "learning_rate" in record
    )
    assert_same_state(uninterrupted_weights, resumed.model.state_dict())
    assert_same_state(uninterrupted_optimizer, resumed.optimizer.state_dict())


def test_incomplete_or_incompatible_resume_is_rejected(tmp_path, tiny_trainer):
    trainer, definition = tiny_trainer(tmp_path, 2)
    trainer.train()
    checkpoint = tmp_path / "checkpoint-2"
    validate_checkpoint(checkpoint, definition, max_steps=3)
    with pytest.raises(ValueError, match="max-steps must exceed"):
        validate_checkpoint(checkpoint, definition, max_steps=2)
    changed = copy.deepcopy(definition)
    changed["training"]["learning_rate"] *= 2
    with pytest.raises(ValueError, match="settings differ"):
        validate_checkpoint(checkpoint, changed, max_steps=3)
    (checkpoint / "optimizer.pt").unlink()
    with pytest.raises(ValueError, match="optimizer.pt"):
        validate_checkpoint(checkpoint, definition, max_steps=3)


def test_training_report_saves_history_and_final_metrics(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import pandas as pd
    import torch
    import transformers

    from post_train_math import rloo
    from post_train_math.cli import build_parser

    frame = pd.DataFrame(
        [
            {
                "problem": "one",
                "type": "Algebra",
                "level": "Level 1",
                "gt_numerator": 1,
                "gt_denominator": 1,
            }
        ]
    )
    monkeypatch.setattr(rloo, "FROZEN_TRAIN_ROWS", 1)
    monkeypatch.setattr(
        rloo,
        "validate_processed_manifest",
        lambda _: (
            {"cohort": {"name": "boxed-numeric"}},
            "data-hash",
        ),
    )
    monkeypatch.setattr(rloo, "load_cohort", lambda *_: frame)
    monkeypatch.setattr(rloo, "validate_model_source", lambda *_: {"pinned": True})
    tokenizer = SimpleNamespace(
        get_chat_template=lambda: "native",
        get_vocab=lambda: {"one": 1},
        special_tokens_map={},
        apply_chat_template=lambda *_args, **_kwargs: {"input_ids": [1, 2]},
        save_pretrained=lambda _: None,
    )
    monkeypatch.setattr(rloo, "load_tokenizer", lambda _: tokenizer)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    original_config = build_config

    def cpu_config(*args, **kwargs):
        return original_config(*args, **kwargs, use_cpu=True)

    monkeypatch.setattr(rloo, "build_config", cpu_config)
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        lambda *_args, **_kwargs: SimpleNamespace(
            config=SimpleNamespace(use_cache=True)
        ),
    )
    history = [{"step": 1, "loss": 0.25}, {"step": 2, "loss": 0.125}]
    metrics = {"train_loss": 0.1875, "train_runtime": 1.5}
    calls = []

    class Trainer:
        def __init__(self, **kwargs):
            self.state = SimpleNamespace(global_step=2, log_history=history)
            assert kwargs["args"].per_device_train_batch_size == 2
            assert kwargs["args"].save_steps == 50

        def train(self, *, resume_from_checkpoint):
            assert resume_from_checkpoint is None
            return SimpleNamespace(metrics=metrics)

        def save_state(self):
            calls.append("state saved")

        def save_model(self, path):
            from pathlib import Path

            Path(path).mkdir(parents=True)

        def is_world_process_zero(self):
            return True

    monkeypatch.setattr(rloo, "resumable_trainer_class", lambda: Trainer)
    args = build_parser().parse_args(
        [
            "train",
            "--model-key",
            "olmo2",
            "--output-dir",
            str(tmp_path / "run"),
            "--prompts-per-update",
            "1",
            "--rollouts",
            "2",
            "--per-device-train-batch-size",
            "2",
            "--gradient-accumulation-steps",
            "1",
            "--max-steps",
            "2",
        ]
    )
    rloo.run_train(args)
    import json

    report = json.loads((args.output_dir / "train_report.json").read_text())
    assert report["log_history"] == history
    assert report["train_metrics"] == metrics
    assert report["global_step"] == 2
    assert report["resume_from_checkpoint"] is None
    assert report["training_config"]["lr_scheduler_type"] == "constant"
    assert calls == ["state saved"]
