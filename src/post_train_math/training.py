"""CLI setup and explicit training settings; the iteration loop lives in rloo.py."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from post_train_math.data import load_cohort, validate_processed_manifest
from post_train_math.models import MODELS, load_tokenizer, validate_model_source
from post_train_math.training_data import (
    FROZEN_TRAIN_ROWS,
    MAX_COMPLETION_LENGTH,
    MAX_PROMPT_LENGTH,
    prepare_training_subset,
)

LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass
class TrainConfig:
    output_dir: Path
    prompts_per_update: int = 8
    num_generations: int = 4
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 32
    world_size: int = 1
    max_steps: int = 200
    max_completion_length: int = MAX_COMPLETION_LENGTH
    learning_rate: float = 1e-5
    warmup_steps: int = 20
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    weight_decay: float = 0.0
    max_grad_norm: float = 0.5
    beta: float = 0.01
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    seed: int = 42
    save_steps: int = 50
    gradient_checkpointing: bool = True
    mixed_precision: str = "no"
    use_cpu: bool = False

    @property
    def generation_batch_size(self):
        return self.prompts_per_update * self.num_generations

    def to_dict(self):
        return {
            **asdict(self),
            "output_dir": str(self.output_dir),
            "generation_batch_size": self.generation_batch_size,
            "num_iterations": 1,
            "sampling": "with replacement; torch seed + iteration",
            "lr_scheduler_type": "linear warmup then constant",
        }


def validate_batch(
    prompts,
    rollouts,
    per_device_train_batch_size,
    world_size,
    gradient_accumulation_steps,
):
    if (
        min(
            prompts,
            per_device_train_batch_size,
            world_size,
            gradient_accumulation_steps,
        )
        < 1
        or rollouts < 2
    ):
        raise ValueError(
            "B, microbatch, world size and accumulation must be positive; G >= 2"
        )
    if prompts * rollouts != (
        per_device_train_batch_size * world_size * gradient_accumulation_steps
    ):
        raise ValueError("B * G must equal microbatch * world-size * accumulation")


def build_config(
    output_dir,
    *,
    prompts,
    rollouts,
    per_device_train_batch_size,
    world_size,
    gradient_accumulation_steps,
    max_steps,
    learning_rate,
    use_cpu=False,
):
    import torch

    validate_batch(
        prompts,
        rollouts,
        per_device_train_batch_size,
        world_size,
        gradient_accumulation_steps,
    )
    if max_steps < 1 or learning_rate <= 0:
        raise ValueError("max-steps and learning-rate must be positive")
    precision = (
        "no" if use_cpu else ("bf16" if torch.cuda.is_bf16_supported() else "fp16")
    )
    return TrainConfig(
        output_dir=Path(output_dir),
        prompts_per_update=prompts,
        num_generations=rollouts,
        per_device_train_batch_size=per_device_train_batch_size,
        world_size=world_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_steps=max_steps,
        learning_rate=learning_rate,
        mixed_precision=precision,
        use_cpu=use_cpu,
    )


def run_train(args):
    resume = args.resume_from_checkpoint
    if resume and args.prepare_only:
        raise ValueError(
            "--resume-from-checkpoint cannot be combined with --prepare-only"
        )
    if (
        args.output_dir.exists()
        and any(args.output_dir.iterdir())
        and (not resume or resume.resolve().parent != args.output_dir.resolve())
    ):
        raise FileExistsError(
            "Use an empty output directory or resume its own checkpoint"
        )

    manifest, manifest_hash = validate_processed_manifest(args.data_dir)
    spec = MODELS[args.model_key]
    frame = load_cohort(args.data_dir, "train")
    if len(frame) != FROZEN_TRAIN_ROWS:
        raise ValueError(
            f"Expected the fixed {FROZEN_TRAIN_ROWS}-question training cohort"
        )
    model_dir = args.model_dir or Path(spec.local_dir)
    if args.prepare_only and args.model_dir is None and not model_dir.exists():
        tokenizer = load_tokenizer(spec.repo_id, revision=spec.revision)
        source = None
    else:
        source = validate_model_source(model_dir, spec)
        tokenizer = load_tokenizer(model_dir)
    if args.prepare_only:
        _, subset = prepare_training_subset(
            frame,
            tokenizer,
            spec,
            manifest_hash,
            args.output_dir,
            write_artifacts=int(os.environ.get("RANK", "0")) == 0,
        )
        print(json.dumps(subset["counts"], indent=2))
        return

    import torch
    from accelerate import Accelerator
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, set_seed

    from post_train_math.checkpoints import run_definition, validate_checkpoint
    from post_train_math.rloo import train_loop

    if not torch.cuda.is_available():
        raise RuntimeError("RLOO training requires CUDA")
    config = build_config(
        args.output_dir,
        prompts=args.prompts_per_update,
        rollouts=args.rollouts,
        per_device_train_batch_size=args.per_device_train_batch_size,
        world_size=int(os.environ.get("WORLD_SIZE", "1")),
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
    )
    accelerator = Accelerator(
        cpu=config.use_cpu,
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=1,
    )
    if accelerator.num_processes != config.world_size:
        raise ValueError("Configured world size does not match Accelerate")
    # DDP synchronizes initial weights; each rank samples independent completions.
    set_seed(config.seed + accelerator.process_index)
    lora = LoraConfig(
        task_type="CAUSAL_LM",
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        target_modules=list(LORA_TARGET_MODULES),
        bias="none",
    )
    definition = run_definition(spec, manifest_hash, config, lora)
    if resume:
        validate_checkpoint(resume, definition, config.max_steps)
    records, subset = prepare_training_subset(
        frame,
        tokenizer,
        spec,
        manifest_hash,
        args.output_dir,
        write_artifacts=accelerator.is_main_process,
    )
    if not records:
        raise ValueError(f"No training prompts fit {MAX_PROMPT_LENGTH} tokens")
    dtype = (
        torch.float32
        if config.use_cpu
        else (torch.bfloat16 if config.mixed_precision == "bf16" else torch.float16)
    )
    model = get_peft_model(
        AutoModelForCausalLM.from_pretrained(
            model_dir,
            local_files_only=True,
            dtype=dtype,
        ),
        lora,
    )
    model.config.use_cache = False
    if config.gradient_checkpointing:
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
    model, optimizer = accelerator.prepare(model, optimizer)
    history = train_loop(
        model,
        tokenizer,
        records,
        optimizer,
        config,
        accelerator,
        definition,
        resume=resume,
    )
    if accelerator.is_main_process:
        report = {
            "model_key": spec.key,
            "model_source": source,
            "data_manifest_sha256": manifest_hash,
            "cohort": manifest["cohort"]["name"],
            "training_subset": subset,
            "training_config": config.to_dict(),
            "global_step": config.max_steps,
            "resume_from_checkpoint": str(resume) if resume else None,
            "log_history": history,
        }
        (args.output_dir / "train_report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
