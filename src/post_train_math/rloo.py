"""RLOO with TRL and LoRA; the frozen base checkpoint is the reference policy."""

from __future__ import annotations

import json
import os
from pathlib import Path

from post_train_math.answers import numeric_reward
from post_train_math.data import load_cohort, validate_processed_manifest
from post_train_math.models import (
    MODELS,
    load_tokenizer,
    make_chat_messages,
    validate_model_source,
)

MAX_PROMPT_LENGTH = 2048
MAX_COMPLETION_LENGTH = 2048


def batch_accumulation(
    prompts: int, rollouts: int, micro_batch: int, world_size: int
) -> int:
    if prompts < 1 or rollouts < 2 or micro_batch < 1 or world_size < 1:
        raise ValueError(
            "B, micro-batch and world size must be positive; R must be >= 2"
        )
    sequences = prompts * rollouts
    if sequences % (micro_batch * world_size):
        raise ValueError("B * R must be divisible by micro-batch * world size")
    return sequences // (micro_batch * world_size)


def build_config(
    output_dir: Path,
    *,
    prompts: int,
    rollouts: int,
    micro_batch: int,
    world_size: int,
    max_steps: int,
    learning_rate: float,
    use_cpu: bool = False,
):
    import torch
    from trl import RLOOConfig

    accumulation = batch_accumulation(prompts, rollouts, micro_batch, world_size)
    if max_steps < 1 or learning_rate <= 0:
        raise ValueError("max-steps and learning-rate must be positive")
    return RLOOConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=micro_batch,
        gradient_accumulation_steps=accumulation,
        generation_batch_size=prompts * rollouts,
        num_generations=rollouts,
        num_iterations=1,
        max_completion_length=MAX_COMPLETION_LENGTH,
        chat_template_kwargs={"truncation": True, "max_length": MAX_PROMPT_LENGTH},
        max_steps=max_steps,
        learning_rate=learning_rate,
        beta=0.05,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        normalize_advantages=False,
        sync_ref_model=False,
        remove_unused_columns=False,
        seed=42,
        data_seed=42,
        use_cpu=use_cpu,
        bf16=not use_cpu and torch.cuda.is_bf16_supported(),
        fp16=not use_cpu and not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        report_to="none",
        logging_steps=1,
        save_strategy="no",
    )


def run_train(args) -> None:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, set_seed
    from trl import RLOOTrainer

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    manifest, manifest_hash = validate_processed_manifest(args.data_dir)
    spec = MODELS[args.model_key]
    model_dir = args.model_dir or Path(spec.local_dir)
    source = validate_model_source(model_dir, spec)
    if not torch.cuda.is_available():
        raise RuntimeError("RLOO training requires CUDA")
    config = build_config(
        args.output_dir,
        prompts=args.prompts_per_update,
        rollouts=args.rollouts,
        micro_batch=args.micro_batch,
        world_size=int(os.environ.get("WORLD_SIZE", "1")),
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
    )
    set_seed(config.seed)
    tokenizer = load_tokenizer(model_dir)
    tokenizer.truncation_side = "left"
    frame = load_cohort(args.data_dir, "train")
    dataset = Dataset.from_list(
        [
            {
                "prompt": make_chat_messages(spec, row.problem),
                "gt_numerator": int(row.gt_numerator),
                "gt_denominator": int(row.gt_denominator),
            }
            for row in frame.itertuples(index=False)
        ]
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        dtype=torch.bfloat16 if config.bf16 else torch.float16,
    )
    model.config.use_cache = False
    trainer = RLOOTrainer(
        model=model,
        args=config,
        processing_class=tokenizer,
        reward_funcs=numeric_reward,
        train_dataset=dataset,
        peft_config=LoraConfig(
            task_type="CAUSAL_LM",
            r=16,
            lora_alpha=32,
            lora_dropout=0.0,
            target_modules="all-linear",
            bias="none",
        ),
    )
    trainer.train()
    trainer.save_model(str(args.output_dir / "adapter"))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(args.output_dir / "adapter")
        report = {
            "model_key": spec.key,
            "model_source": source,
            "data_manifest_sha256": manifest_hash,
            "cohort": manifest["cohort"]["name"],
            "train_rows": len(frame),
            "prompts_per_update": args.prompts_per_update,
            "rollouts_per_prompt": args.rollouts,
            "max_prompt_length": MAX_PROMPT_LENGTH,
            "prompt_overflow": "left truncation; no rows removed",
            "max_completion_length": MAX_COMPLETION_LENGTH,
            "reference_policy": "pinned base with LoRA disabled",
            "training_config": config.to_dict(),
        }
        (args.output_dir / "train_report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
