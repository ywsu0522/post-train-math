import json

import pytest

from post_train_math.answers import numeric_reward
from post_train_math.rloo import batch_accumulation, build_config


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
    assert batch_accumulation(8, 4, 1, world_size) == expected


@pytest.mark.parametrize("batch", [(0, 4, 1, 1), (8, 1, 1, 1), (3, 2, 4, 1)])
def test_invalid_batches_are_rejected(batch):
    with pytest.raises(ValueError):
        batch_accumulation(*batch)


def test_real_trl_peft_cpu_step_and_adapter_save(tmp_path, monkeypatch):
    """Exercise the locked TRL API locally without network or model weights."""
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
        micro_batch=1,
        world_size=1,
        max_steps=1,
        learning_rate=1e-3,
        use_cpu=True,
    )
    assert config.num_generations == 2
    assert config.generation_batch_size == 4
    assert config.gradient_accumulation_steps == 4
    assert config.max_completion_length == 2048
    assert config.chat_template_kwargs == {"truncation": True, "max_length": 2048}
    assert config.num_iterations == 1 and not config.sync_ref_model
    config.max_completion_length = 4
    config.disable_tqdm = True
    dataset = Dataset.from_list(
        [
            {
                "prompt": [{"role": "user", "content": prompt}],
                "gt_numerator": answer,
                "gt_denominator": 1,
            }
            for prompt, answer in [("Problem one", 1), ("Problem two", 2)]
        ]
    )
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
    trainer.train()
    assert trainer.state.global_step == 1
    for name, param in trainer.model.named_parameters():
        if name in frozen:
            assert torch.equal(param.detach(), frozen[name])
    trainer.save_model(str(tmp_path / "adapter"))
    saved = json.loads((tmp_path / "adapter/adapter_config.json").read_text())
    assert saved["peft_type"] == "LORA"
