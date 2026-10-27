"""RLOO with TRL and LoRA; the frozen base checkpoint is the reference policy."""

from __future__ import annotations

import hashlib
import json
import os
from importlib.metadata import version
from pathlib import Path

from post_train_math.answers import numeric_reward
from post_train_math.data import COHORT_NAME, load_cohort, validate_processed_manifest
from post_train_math.models import (
    MODELS,
    load_tokenizer,
    make_chat_messages,
    validate_model_source,
)

MAX_PROMPT_LENGTH = 2048
MAX_COMPLETION_LENGTH = 2048
FROZEN_TRAIN_ROWS = 5589
CHECKPOINT_STEPS = 50


def resumable_trainer_class():
    """Make the repeat sampler reproducible from the checkpoint's epoch."""
    from torch.utils.data import Sampler
    from trl import RLOOTrainer

    class EpochSampler(Sampler):
        def __init__(self, sampler):
            self.sampler = sampler
            self.epoch = 0

        def set_epoch(self, epoch):
            self.epoch = epoch

        def __len__(self):
            return len(self.sampler)

        def __iter__(self):
            if self.sampler.shuffle:
                self.sampler.generator.manual_seed(self.sampler.seed + self.epoch)
            yield from self.sampler

    class ResumableRLOOTrainer(RLOOTrainer):
        def _get_train_sampler(self, dataset=None):
            return EpochSampler(super()._get_train_sampler(dataset))

    return ResumableRLOOTrainer


def checkpoint_callback(definition: dict):
    """Bind each complete Trainer checkpoint to its data and training settings."""
    from transformers import TrainerCallback

    class CheckpointDefinitionCallback(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            if state.is_world_process_zero:
                checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
                (checkpoint / "run_definition.json").write_text(
                    json.dumps(definition, sort_keys=True, indent=2) + "\n",
                    encoding="utf-8",
                )

    return CheckpointDefinitionCallback()


def validate_checkpoint(checkpoint: Path, definition: dict, *, max_steps: int) -> None:
    """Never silently fall back to a weights-only or incompatible restart."""
    required = [
        "adapter_config.json",
        "optimizer.pt",
        "scheduler.pt",
        "trainer_state.json",
        "training_args.bin",
        "run_definition.json",
    ]
    world_size = definition["world_size"]
    required += (
        ["rng_state.pth"]
        if world_size == 1
        else [f"rng_state_{rank}.pth" for rank in range(world_size)]
    )
    if definition["training"]["fp16"]:
        required.append("scaler.pt")
    missing = [name for name in required if not (checkpoint / name).is_file()]
    if not any(
        (checkpoint / name).is_file()
        for name in ("adapter_model.safetensors", "adapter_model.bin")
    ):
        missing.append("adapter_model.safetensors or adapter_model.bin")
    if missing:
        raise ValueError(
            f"Incomplete training checkpoint {checkpoint}: missing {', '.join(missing)}"
        )
    saved = json.loads((checkpoint / "run_definition.json").read_text(encoding="utf-8"))
    if saved != definition:
        raise ValueError(
            "Checkpoint model, data, subset, world size or training settings differ"
        )
    state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    if not 0 < state["global_step"] < max_steps:
        raise ValueError(
            "max-steps must exceed the checkpoint global_step (total target, not extra steps)"
        )


def run_definition(spec, manifest_hash: str, subset: dict, config, lora) -> dict:
    """Settings that must remain identical; max_steps may be extended with constant LR."""
    keys = (
        "learning_rate",
        "lr_scheduler_type",
        "warmup_steps",
        "optim",
        "adam_beta1",
        "adam_beta2",
        "adam_epsilon",
        "weight_decay",
        "max_grad_norm",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "generation_batch_size",
        "num_generations",
        "num_iterations",
        "max_completion_length",
        "chat_template_kwargs",
        "beta",
        "temperature",
        "top_p",
        "top_k",
        "normalize_advantages",
        "sync_ref_model",
        "seed",
        "data_seed",
        "bf16",
        "fp16",
        "gradient_checkpointing",
    )
    values = config.to_dict()
    return {
        "model": {"repo_id": spec.repo_id, "revision": spec.revision},
        "data_manifest_sha256": manifest_hash,
        "training_subset": subset,
        "world_size": config.world_size,
        "sampler_seed_rule": "seed + epoch",
        "training": {key: values[key] for key in keys},
        "lora": lora.to_dict(),
        "versions": {
            package: version(package)
            for package in ("torch", "trl", "peft", "transformers")
        },
    }


def validate_batch(
    prompts: int,
    rollouts: int,
    per_device_train_batch_size: int,
    world_size: int,
    gradient_accumulation_steps: int,
) -> None:
    if (
        prompts < 1
        or rollouts < 2
        or per_device_train_batch_size < 1
        or world_size < 1
        or gradient_accumulation_steps < 1
    ):
        raise ValueError(
            "B, per-device-train-batch-size, world size and accumulation must be positive; R must be >= 2"
        )
    if (
        prompts * rollouts
        != per_device_train_batch_size * world_size * gradient_accumulation_steps
    ):
        raise ValueError(
            "B * R must equal per-device-train-batch-size * world-size * gradient-accumulation-steps "
            "to keep one optimizer update per rollout batch"
        )


def _json_bytes(value) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def prepare_training_subset(
    frame,
    tokenizer,
    spec,
    data_manifest_hash: str,
    output_dir: Path,
    *,
    write_artifacts: bool = True,
) -> tuple[list[dict], dict]:
    """Audit every frozen train question using the same native template as TRL."""
    template = tokenizer.get_chat_template()
    selected, audit, overlong = [], [], []
    seen = set()
    for row in frame.itertuples(index=False):
        problem_id = "math-" + hashlib.sha256(row.problem.encode("utf-8")).hexdigest()
        if problem_id in seen:
            raise ValueError(f"Duplicate train problem ID: {problem_id}")
        seen.add(problem_id)
        prompt = make_chat_messages(spec, row.problem)
        encoded = tokenizer.apply_chat_template(
            prompt,
            tokenize=True,
            add_generation_prompt=True,
            truncation=False,
            padding=False,
            return_dict=True,
        )
        token_ids = encoded["input_ids"]
        prompt_tokens = len(token_ids)
        included = prompt_tokens <= MAX_PROMPT_LENGTH
        entry = {
            "problem_id": problem_id,
            "prompt_tokens": prompt_tokens,
            "type": row.type,
            "level": row.level,
            "included": included,
            "prompt_token_ids_sha256": hashlib.sha256(
                _json_bytes(token_ids)
            ).hexdigest(),
        }
        audit.append(entry)
        if included:
            selected.append(
                {
                    **entry,
                    "prompt": prompt,
                    "gt_numerator": int(row.gt_numerator),
                    "gt_denominator": int(row.gt_denominator),
                }
            )
        else:
            overlong.append(entry)
    # Record the actual tokenizer state as well as the pinned checkpoint identity.
    backend = getattr(tokenizer, "backend_tokenizer", None)
    tokenizer_state = (
        backend.to_str().encode("utf-8")
        if backend is not None
        else _json_bytes(tokenizer.get_vocab())
    )
    artifacts = {
        "training_subset.jsonl": selected,
        "prompt_audit.jsonl": audit,
        "overlong_prompts.jsonl": overlong,
    }
    payloads = {
        name: b"".join(_json_bytes(row) + b"\n" for row in rows)
        for name, rows in artifacts.items()
    }
    definition = {
        "cohort": COHORT_NAME,
        "split": "train",
        "data_manifest_sha256": data_manifest_hash,
        "problem_id_definition": "math- + SHA256 of the original problem text encoded as UTF-8",
        "model": {"key": spec.key, "repo_id": spec.repo_id, "revision": spec.revision},
        "tokenization": {
            "chat_template": template,
            "chat_template_sha256": hashlib.sha256(
                template.encode("utf-8")
            ).hexdigest(),
            "system_prompt": spec.system_prompt,
            "tokenizer_class": type(tokenizer).__name__,
            "tokenizer_state_sha256": hashlib.sha256(tokenizer_state).hexdigest(),
            "special_tokens_map": tokenizer.special_tokens_map,
            "add_generation_prompt": True,
            "truncation": False,
            "padding": False,
            "transformers_version": version("transformers"),
            "tokenizers_version": version("tokenizers"),
        },
        "selection": {
            "field": "prompt_tokens",
            "operator": "<=",
            "threshold": MAX_PROMPT_LENGTH,
        },
        "counts": {
            "cohort_train": len(audit),
            "selected": len(selected),
            "overlong": len(overlong),
        },
        "files_sha256": {
            name: hashlib.sha256(payload).hexdigest()
            for name, payload in payloads.items()
        },
    }
    if write_artifacts:
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, payload in payloads.items():
            (output_dir / name).write_bytes(payload)
        (output_dir / "training_subset_manifest.json").write_bytes(
            json.dumps(definition, ensure_ascii=False, sort_keys=True, indent=2).encode(
                "utf-8"
            )
            + b"\n"
        )
    return selected, definition


def build_config(
    output_dir: Path,
    *,
    prompts: int,
    rollouts: int,
    per_device_train_batch_size: int,
    world_size: int,
    gradient_accumulation_steps: int,
    max_steps: int,
    learning_rate: float,
    use_cpu: bool = False,
):
    import torch
    from trl import RLOOConfig

    validate_batch(
        prompts,
        rollouts,
        per_device_train_batch_size,
        world_size,
        gradient_accumulation_steps,
    )
    if max_steps < 1 or learning_rate <= 0:
        raise ValueError("max-steps and learning-rate must be positive")
    return RLOOConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        generation_batch_size=prompts * rollouts,
        num_generations=rollouts,
        num_iterations=1,
        max_completion_length=MAX_COMPLETION_LENGTH,
        chat_template_kwargs={"truncation": False},
        max_steps=max_steps,
        learning_rate=learning_rate,
        lr_scheduler_type="constant",
        warmup_steps=0,
        optim="adamw_torch",
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
        save_strategy="steps",
        save_steps=CHECKPOINT_STEPS,
        save_total_limit=None,
        save_only_model=False,
        restore_callback_states_from_checkpoint=True,
        ignore_data_skip=False,
    )


def run_train(args) -> None:
    resume = args.resume_from_checkpoint
    if resume and args.prepare_only:
        raise ValueError(
            "--resume-from-checkpoint cannot be combined with --prepare-only"
        )
    if not resume and args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    if (
        resume
        and args.output_dir.exists()
        and any(args.output_dir.iterdir())
        and resume.resolve().parent != args.output_dir.resolve()
    ):
        raise ValueError(
            "A nonempty output-dir can only resume one of its own checkpoints"
        )
    manifest, manifest_hash = validate_processed_manifest(args.data_dir)
    spec = MODELS[args.model_key]
    model_dir = args.model_dir or Path(spec.local_dir)
    frame = load_cohort(args.data_dir, "train")
    if len(frame) != FROZEN_TRAIN_ROWS:
        raise ValueError(
            f"Expected the fixed {FROZEN_TRAIN_ROWS}-question boxed-numeric train cohort"
        )
    if args.prepare_only and args.model_dir is None and not model_dir.exists():
        # CPU audit needs only the tokenizer at the exact pinned commit, not weights.
        tokenizer = load_tokenizer(spec.repo_id, revision=spec.revision)
        source = None
    else:
        source = validate_model_source(model_dir, spec)
        tokenizer = load_tokenizer(model_dir)
    if args.prepare_only:
        is_main_process = int(os.environ.get("RANK", "0")) == 0
        _, subset = prepare_training_subset(
            frame,
            tokenizer,
            spec,
            manifest_hash,
            args.output_dir,
            write_artifacts=is_main_process,
        )
        if is_main_process:
            print(json.dumps(subset["counts"], indent=2))
        return

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, set_seed

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
    set_seed(config.seed)
    records, subset = prepare_training_subset(
        frame,
        tokenizer,
        spec,
        manifest_hash,
        args.output_dir,
        write_artifacts=False,
    )
    if not records:
        raise ValueError(
            "No training prompts fit the 2048-token budget; see prompt_audit.jsonl"
        )
    lora = LoraConfig(
        task_type="CAUSAL_LM",
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        target_modules="all-linear",
        bias="none",
    )
    definition = run_definition(spec, manifest_hash, subset, config, lora)
    if resume:
        validate_checkpoint(resume, definition, max_steps=args.max_steps)
    # Validate the previous run before writing any artifacts into its directory.
    if config.process_index == 0:
        prepare_training_subset(frame, tokenizer, spec, manifest_hash, args.output_dir)
    dataset = Dataset.from_list(records)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        dtype=torch.bfloat16 if config.bf16 else torch.float16,
    )
    model.config.use_cache = False
    trainer = resumable_trainer_class()(
        model=model,
        args=config,
        processing_class=tokenizer,
        reward_funcs=numeric_reward,
        train_dataset=dataset,
        peft_config=lora,
        callbacks=[checkpoint_callback(definition)],
    )
    result = trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    trainer.save_state()
    trainer.save_model(str(args.output_dir / "adapter"))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(args.output_dir / "adapter")
        report = {
            "model_key": spec.key,
            "model_source": source,
            "data_manifest_sha256": manifest_hash,
            "cohort": manifest["cohort"]["name"],
            "cohort_train_rows": len(frame),
            "train_rows": len(records),
            "training_subset": subset,
            "training_subset_manifest_sha256": hashlib.sha256(
                (args.output_dir / "training_subset_manifest.json").read_bytes()
            ).hexdigest(),
            "prompts_per_update": args.prompts_per_update,
            "rollouts_per_prompt": args.rollouts,
            "max_prompt_length": MAX_PROMPT_LENGTH,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "prompt_overflow": "exclude prompt_tokens > 2048; no truncation",
            "max_completion_length": MAX_COMPLETION_LENGTH,
            "reference_policy": "pinned base with LoRA disabled",
            "training_config": config.to_dict(),
            "resume_from_checkpoint": str(resume.resolve()) if resume else None,
            "global_step": trainer.state.global_step,
            "log_history": trainer.state.log_history,
            "train_metrics": result.metrics,
        }
        (args.output_dir / "train_report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
