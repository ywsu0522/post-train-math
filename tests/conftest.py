"""Small real PEFT models for exercising the explicit RLOO loop offline."""

import pytest


@pytest.fixture
def tiny_training(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    import torch
    from accelerate import Accelerator
    from peft import LoraConfig, get_peft_model
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from transformers import (
        GraniteConfig,
        GraniteForCausalLM,
        Olmo2Config,
        Olmo2ForCausalLM,
        PreTrainedTokenizerFast,
        set_seed,
    )

    from post_train_math.checkpoints import run_definition
    from post_train_math.models import MODELS
    from post_train_math.training import LORA_TARGET_MODULES, build_config

    threads = torch.get_num_threads()
    torch.set_num_threads(1)

    def make(output, max_steps=3, model_key="olmo2", microbatch=2):
        set_seed(42)
        backend = Tokenizer(
            WordLevel(
                {
                    "[PAD]": 0,
                    "[EOS]": 1,
                    "[UNK]": 2,
                    "one": 3,
                    "two": 4,
                    "answer": 5,
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
        )
        tokenizer.chat_template = (
            "{% for m in messages %}{{ m['content'] }} {% endfor %}answer"
        )
        config = build_config(
            output,
            prompts=2,
            rollouts=2,
            per_device_train_batch_size=microbatch,
            world_size=1,
            gradient_accumulation_steps=4 // microbatch,
            max_steps=max_steps,
            learning_rate=1e-3,
            use_cpu=True,
        )
        config.max_completion_length = 3
        config.save_steps = 2
        model_config, model_class = {
            "olmo2": (Olmo2Config, Olmo2ForCausalLM),
            "granite": (GraniteConfig, GraniteForCausalLM),
        }[model_key]
        base = model_class(
            model_config(
                vocab_size=8,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
                max_position_embeddings=32,
                eos_token_id=1,
                pad_token_id=0,
                use_cache=False,
            )
        )
        lora = LoraConfig(
            task_type="CAUSAL_LM",
            r=2,
            lora_alpha=4,
            target_modules=list(LORA_TARGET_MODULES),
            lora_dropout=0.0,
        )
        model = get_peft_model(base, lora)
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_epsilon,
            weight_decay=config.weight_decay,
        )
        accelerator = Accelerator(cpu=True)
        model, optimizer = accelerator.prepare(model, optimizer)
        records = [
            {
                "prompt": [{"role": "user", "content": text}],
                "problem_id": str(i),
                "gt_numerator": 1,
                "gt_denominator": 1,
            }
            for i, text in enumerate(("one", "two", "one two", "two one"))
        ]
        definition = run_definition(MODELS[model_key], "data-hash", config, lora)
        config.output_dir.mkdir(parents=True, exist_ok=True)
        return model, tokenizer, records, optimizer, config, accelerator, definition

    yield make
    torch.set_num_threads(threads)
