import json
from pathlib import Path

import pytest

from post_train_math.answers import numeric_reward
from post_train_math.rloo import build_config, prepare_training_subset, validate_batch


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


def test_real_trl_peft_cpu_step_and_adapter_save(tmp_path, monkeypatch):
    """Exercise the locked TRL API locally without network or model weights."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    monkeypatch.setenv("TRL_HUB_TELEMETRY", "0")
    import pandas as pd
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
    from trl import RLOOTrainer

    set_seed(42)
    backend = Tokenizer(
        WordLevel(
            {
                "[PAD]": 0,
                "[EOS]": 1,
                "[UNK]": 2,
                "Problem": 3,
                "one": 4,
                "two": 5,
                r"\boxed{1}": 6,
                r"\boxed{2}": 7,
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
        truncation_side="left",
    )
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['content'] }} {% endfor %}"
    )
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=8,
            n_layer=1,
            n_head=1,
            n_embd=8,
            n_positions=32,
            bos_token_id=1,
            eos_token_id=1,
            pad_token_id=0,
            use_cache=False,
        )
    )
    config = build_config(
        tmp_path,
        prompts=2,
        rollouts=2,
        per_device_train_batch_size=1,
        world_size=1,
        gradient_accumulation_steps=4,
        max_steps=1,
        learning_rate=1e-3,
        use_cpu=True,
    )
    assert config.num_generations == 2
    assert config.generation_batch_size == 4
    assert config.gradient_accumulation_steps == 4
    assert config.max_completion_length == 2048
    assert config.chat_template_kwargs == {"truncation": False}
    assert config.num_iterations == 1 and not config.sync_ref_model
    assert config.lr_scheduler_type == "constant"
    assert config.optim == "adamw_torch"
    assert config.warmup_steps == 0
    assert config.save_strategy == "steps" and config.save_steps == 50
    assert not config.save_only_model
    config.max_completion_length = 4
    config.disable_tqdm = True
    from dataclasses import replace

    from post_train_math.models import MODELS

    frame = pd.DataFrame(
        [
            {
                "problem": prompt,
                "type": "Algebra",
                "level": "Level 1",
                "gt_numerator": answer,
                "gt_denominator": 1,
            }
            for prompt, answer in [
                ("Problem one", 1),
                ("Problem two", 2),
                ("one " * 2050, 1),
            ]
        ]
    )
    records, subset = prepare_training_subset(
        frame,
        tokenizer,
        replace(MODELS["olmo2"], system_prompt="one"),
        "fixed-data-hash",
        tmp_path / "subset",
    )
    assert subset["counts"] == {"cohort_train": 3, "selected": 2, "overlong": 1}
    dataset = Dataset.from_list(records)
    trainer = RLOOTrainer(
        model=model,
        args=config,
        processing_class=tokenizer,
        train_dataset=dataset,
        reward_funcs=numeric_reward,
        peft_config=LoraConfig(
            task_type="CAUSAL_LM",
            r=2,
            lora_alpha=4,
            target_modules="all-linear",
            lora_dropout=0.0,
        ),
    )
    frozen = {
        name: param.detach().clone()
        for name, param in trainer.model.named_parameters()
        if not param.requires_grad
    }
    assert all(
        "lora_" in name
        for name, param in trainer.model.named_parameters()
        if param.requires_grad
    )
    actual_ids, _, _ = trainer._tokenize_prompts(
        [record["prompt"] for record in records]
    )
    assert [len(ids) for ids in actual_ids] == [
        record["prompt_tokens"] for record in records
    ]
    trainer.train()
    assert trainer.state.global_step == 1
    for name, param in trainer.model.named_parameters():
        if name in frozen:
            assert torch.equal(param.detach(), frozen[name])
    trainer.save_model(str(tmp_path / "adapter"))
    saved = json.loads((tmp_path / "adapter/adapter_config.json").read_text())
    assert saved["peft_type"] == "LORA"


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
    assert definition["counts"]["cohort_train"] == 5029
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
    assert len(audit) == len({record["problem_id"] for record in audit}) == 5029
    assert {record["problem_id"] for record in selected} == {
        record["problem_id"] for record in audit if record["prompt_tokens"] <= 2048
    }
    assert len(selected) == definition["counts"]["selected"]
