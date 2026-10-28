"""Shared explicit policy-gradient engine, independent of individual RL methods.

Read train_loop top to bottom: generate_rollouts, method.estimate_gradients,
then update_parameters. A method returns only after LoRA .grad is ready.
"""

from __future__ import annotations

import json
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence

from post_train_math.answers import numeric_reward
from post_train_math.checkpoints import load_checkpoint, save_checkpoint

LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


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


@dataclass
class Rollouts:
    records: list[dict]
    prompt_ids: list[list[int]]
    completion_ids: list[list[int]]
    texts: list[str]
    sampled_indices: list[int]


@dataclass
class GradientContext:
    model: Any
    tokenizer: Any
    optimizer: Any
    config: Any
    accelerator: Any
    iteration: int
    trainable: dict

    def local_advantages(self, advantages, count):
        start = self.accelerator.process_index * count
        return advantages[start : start + count]


@dataclass
class CompletionLogProbs:
    values: torch.Tensor
    mask: torch.Tensor

    @property
    def sums(self):
        return (self.values * self.mask).sum(1)

    @property
    def lengths(self):
        return self.mask.sum(1)

    @property
    def means(self):
        return self.sums / self.lengths.clamp_min(1)


@dataclass
class LossTerms:
    # Contributions must sum over microbatches, then average over DDP ranks,
    # to recover the intended global rollout objective.
    loss: torch.Tensor
    metrics: dict[str, torch.Tensor] = field(default_factory=dict)


def log_stage(accelerator, output_dir, iteration, stage, **values):
    """Edit here, or add print/log statements directly around the three stages."""
    if accelerator.is_main_process:
        event = {"iteration": iteration, "stage": stage, **values}
        line = json.dumps(event, ensure_ascii=False, allow_nan=False)
        print(line, flush=True)
        with (Path(output_dir) / "iteration_log.jsonl").open(
            "a", encoding="utf-8"
        ) as f:
            f.write(line + "\n")


def train_loop(
    model,
    tokenizer,
    records,
    optimizer,
    config,
    accelerator,
    definition,
    *,
    method,
    resume=None,
    reward_fn=numeric_reward,
):
    """Visible orchestration: rollout -> method gradients -> parameter update."""
    if method.NAME != config.method:
        raise ValueError("Selected method and training configuration disagree")
    if (config.temperature, config.top_p, config.top_k) != (1.0, 1.0, 0):
        raise ValueError("Raw-policy gradients require temperature=1, top_p=1, top_k=0")
    validate_batch(
        config.prompts_per_update,
        config.num_generations,
        config.per_device_train_batch_size,
        accelerator.num_processes,
        config.gradient_accumulation_steps,
    )
    if not records:
        raise ValueError("Training cohort is empty")
    if accelerator.gradient_accumulation_steps != 1:
        raise ValueError(
            "This loop accumulates explicitly; Accelerate's factor must be 1"
        )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    trainable = verify_lora_targets(model)
    optimizer_parameters = {
        id(p) for group in optimizer.param_groups for p in group["params"]
    }
    if optimizer_parameters != {id(p) for p in trainable.values()}:
        raise ValueError("AdamW must contain exactly the trainable LoRA parameters")
    step, optimizer_steps, history = (
        (0, 0, [])
        if resume is None
        else load_checkpoint(
            resume, model, optimizer, accelerator, definition, config.max_steps
        )
    )
    log_stage(
        accelerator,
        config.output_dir,
        step,
        "setup",
        method=method.NAME,
        lora_targets=list(LORA_TARGET_MODULES),
        lora_parameter_tensors=len(trainable),
        trainable_parameters=sum(p.numel() for p in trainable.values()),
        rollout_iterations=step,
        optimizer_steps=optimizer_steps,
    )
    for iteration in range(step + 1, config.max_steps + 1):
        # Stage 1.1: sample B prompts; PEFT generate -> BG completions.
        log_stage(accelerator, config.output_dir, iteration, "1.1", status="start")
        rollouts = generate_rollouts(
            model, tokenizer, records, config, accelerator, iteration
        )
        log_stage(
            accelerator,
            config.output_dir,
            iteration,
            "1.1",
            status="complete",
            sampled_indices=rollouts.sampled_indices,
            sampled_problem_ids=[
                records[i]["problem_id"] for i in rollouts.sampled_indices
            ],
            prompts=config.prompts_per_update,
            completions=config.generation_batch_size,
        )

        # Stage 1.2: the selected RL method must finish accumulating LoRA .grad.
        log_stage(accelerator, config.output_dir, iteration, "1.2", status="start")
        context = GradientContext(
            model, tokenizer, optimizer, config, accelerator, iteration, trainable
        )
        gradient_stats = method.estimate_gradients(
            context, rollouts, reward_fn=reward_fn
        )
        if any(parameter.grad is None for parameter in trainable.values()):
            raise RuntimeError(
                f"{method.NAME}.estimate_gradients returned without ready LoRA gradients"
            )
        log_stage(
            accelerator,
            config.output_dir,
            iteration,
            "1.2",
            status="complete",
            **gradient_stats,
        )

        # Stage 2: AdamW consumes the accumulated .grad; verify LoRA changed.
        log_stage(accelerator, config.output_dir, iteration, "2", status="start")
        update_stats = update_parameters(
            optimizer,
            config,
            accelerator,
            optimizer_step=optimizer_steps + 1,
            trainable=trainable,
        )
        optimizer_steps += 1
        progress = {
            "rollout_iterations": iteration,
            "optimizer_steps": optimizer_steps,
        }
        log_stage(
            accelerator,
            config.output_dir,
            iteration,
            "2",
            status="complete",
            **update_stats,
            **progress,
        )
        history.append(
            {"step": iteration, **progress, **gradient_stats, **update_stats}
        )

        if iteration % config.save_steps == 0 or iteration == config.max_steps:
            save_checkpoint(
                config.output_dir / f"checkpoint-{iteration}",
                model,
                tokenizer,
                optimizer,
                accelerator,
                definition,
                iteration,
                history,
                optimizer_steps=optimizer_steps,
            )
    save_checkpoint(
        config.output_dir / "adapter",
        model,
        tokenizer,
        optimizer,
        accelerator,
        definition,
        config.max_steps,
        history,
        optimizer_steps=optimizer_steps,
        weights_only=True,
    )
    return history


def generate_rollouts(model, tokenizer, records, config, accelerator, iteration):
    """Stage 1.1: sample global B prompts WITH replacement, generate B*G answers."""
    generator = torch.Generator().manual_seed(config.seed + iteration)
    indices = torch.randint(
        len(records), (config.prompts_per_update,), generator=generator
    ).tolist()
    expanded = [records[i] for i in indices for _ in range(config.num_generations)]
    count = len(expanded) // accelerator.num_processes
    start = accelerator.process_index * count
    local_records = expanded[start : start + count]
    policy = accelerator.unwrap_model(model)
    policy.eval()
    prompt_ids, completion_ids = [], []
    eos = policy.generation_config.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]
    eos = [token for token in eos if token is not None]
    for start in range(0, count, config.per_device_train_batch_size):
        rows = local_records[start : start + config.per_device_train_batch_size]
        inputs = tokenizer.apply_chat_template(
            [row["prompt"] for row in rows],
            tokenize=True,
            add_generation_prompt=True,
            truncation=False,
            padding=True,
            return_tensors="pt",
            return_dict=True,
        ).to(accelerator.device)
        with torch.no_grad(), accelerator.autocast():
            generated = policy.generate(
                **inputs,
                do_sample=True,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                max_new_tokens=config.max_completion_length,
                pad_token_id=tokenizer.pad_token_id,
            )
        completions = generated[:, inputs["input_ids"].size(1) :]
        # Retain the first EOS itself, discard tokens/padding after it.
        is_eos = torch.isin(completions, torch.tensor(eos, device=completions.device))
        mask = is_eos.long().cumsum(1) - is_eos.long() == 0
        prompt_ids.extend(
            ids[valid.bool()].tolist()
            for ids, valid in zip(
                inputs["input_ids"], inputs["attention_mask"], strict=True
            )
        )
        completion_ids.extend(
            ids[valid].tolist() for ids, valid in zip(completions, mask, strict=True)
        )
    if len(completion_ids) != count or any(not ids for ids in completion_ids):
        raise RuntimeError(
            "Generation must return one nonempty completion per sequence"
        )
    return Rollouts(
        local_records,
        prompt_ids,
        completion_ids,
        tokenizer.batch_decode(completion_ids, skip_special_tokens=True),
        indices,
    )


def completion_log_probs(model, tokenizer, rollouts, start, end, device):
    """Differentiable completion-token log-probs and mask, including first EOS."""
    prompts = [
        torch.tensor(ids, device=device) for ids in rollouts.prompt_ids[start:end]
    ]
    answers = [
        torch.tensor(ids, device=device) for ids in rollouts.completion_ids[start:end]
    ]
    prompt_ids = pad_sequence(
        prompts,
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
        padding_side="left",
    )
    answer_ids = pad_sequence(
        answers, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    prompt_mask = pad_sequence(
        [torch.ones_like(ids) for ids in prompts], batch_first=True, padding_side="left"
    )
    answer_mask = pad_sequence(
        [torch.ones_like(ids) for ids in answers], batch_first=True
    )
    attention_mask = torch.cat([prompt_mask, answer_mask], dim=1)
    outputs = model(
        input_ids=torch.cat([prompt_ids, answer_ids], dim=1),
        attention_mask=attention_mask,
        position_ids=(attention_mask.cumsum(1) - 1).clamp(min=0),
        logits_to_keep=answer_ids.size(1) + 1,
        use_cache=False,
    )
    logits = outputs.logits[:, :-1].float()
    token_logps = logits.gather(2, answer_ids.unsqueeze(-1)).squeeze(-1)
    token_logps = token_logps - logits.logsumexp(-1)
    return CompletionLogProbs(token_logps, answer_mask)


def sequence_log_probs(model, tokenizer, rollouts, start, end, device):
    """Sequence-sum view of the shared token log-probability computation."""
    return completion_log_probs(model, tokenizer, rollouts, start, end, device).sums


def score_rollouts(context, rollouts, reward_fn):
    rewards = torch.tensor(
        reward_fn(
            completions=rollouts.texts,
            gt_numerator=[row["gt_numerator"] for row in rollouts.records],
            gt_denominator=[row["gt_denominator"] for row in rollouts.records],
        ),
        device=context.accelerator.device,
        dtype=torch.float32,
    )
    if rewards.shape != (len(rollouts.records),) or not torch.isfinite(rewards).all():
        raise ValueError("Verifier must return one finite reward per completion")
    return rewards


def log_advantages(context, advantages, **values):
    log_stage(
        context.accelerator,
        context.config.output_dir,
        context.iteration,
        "1.2",
        status="advantages",
        advantages=advantages.reshape(-1, context.config.num_generations).tolist(),
        advantage_min=advantages.min().item(),
        advantage_max=advantages.max().item(),
        advantage_mean=advantages.mean().item(),
        advantage_std=advantages.std(unbiased=False).item(),
        **values,
    )


def accumulate_gradients(context, rollouts, loss_fn):
    """Common microbatch backward/DDP/AMP; leave unscaled, audited .grad ready.

    loss_fn receives (CompletionLogProbs, start, end) and returns LossTerms.
    It owns mathematical normalization; this engine never averages microbatch
    means or applies a second accumulation factor.
    """
    model, accelerator = context.model, context.accelerator
    count = len(rollouts.records)
    micro = context.config.per_device_train_batch_size
    context.optimizer.zero_grad(set_to_none=True)
    model.train()
    loss_value = 0.0
    metric_values = {}
    gradient_logps = []
    for start in range(0, count, micro):
        end = min(start + micro, count)
        sync = nullcontext() if end == count else accelerator.no_sync(model)
        with sync:
            with accelerator.autocast():
                logps = completion_log_probs(
                    model, context.tokenizer, rollouts, start, end, accelerator.device
                )
                if not logps.values.requires_grad:
                    raise RuntimeError(
                        "Policy log probabilities must retain the LoRA graph"
                    )
                gradient_logps.append(logps.sums.detach())
                terms = loss_fn(logps, start, end)
                if not torch.isfinite(terms.loss):
                    raise RuntimeError("Non-finite policy loss; training aborted")
            accelerator.backward(terms.loss)
        loss_value += terms.loss.detach().item()
        for name, value in terms.metrics.items():
            metric_values[name] = metric_values.get(name, 0.0) + value.detach().item()
    accelerator.unscale_gradients(context.optimizer)
    stats = gradient_statistics(model, context.trainable)
    for name, value in {"loss": loss_value, **metric_values}.items():
        stats[name] = accelerator.reduce(
            torch.tensor(value, device=accelerator.device), reduction="mean"
        ).item()
    stats["policy_log_probs"] = accelerator.gather(torch.cat(gradient_logps)).tolist()
    stats["policy_log_probs_require_grad"] = logps.values.requires_grad
    return stats


def verify_lora_targets(model):
    """Fail before training if any base/head parameter is trainable."""
    trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
    targets = set()
    for name in trainable:
        if not any(part in name for part in (".lora_A.", ".lora_B.")):
            raise ValueError(f"Non-LoRA trainable parameter: {name}")
        target = name.split(".lora_")[0].rsplit(".", 1)[-1]
        if target not in LORA_TARGET_MODULES:
            raise ValueError(f"Unexpected LoRA target: {name}")
        targets.add(target)
    if targets != set(LORA_TARGET_MODULES):
        raise ValueError(f"Missing LoRA targets: {set(LORA_TARGET_MODULES) - targets}")
    return trainable


def gradient_statistics(model, trainable):
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad and parameter.grad is not None:
            raise RuntimeError(f"Frozen base parameter received gradient: {name}")
    by_target = {target: [] for target in LORA_TARGET_MODULES}
    for name, parameter in trainable.items():
        if parameter.grad is None:
            raise RuntimeError(f"LoRA parameter has no gradient: {name}")
        norm = parameter.grad.detach().float().norm()
        if not torch.isfinite(norm):
            raise RuntimeError(f"Non-finite LoRA gradient: {name}")
        by_target[name.split(".lora_")[0].rsplit(".", 1)[-1]].append(norm)
    return {
        "lora_gradient_norm_by_target": {
            name: torch.stack(norms).norm().item() for name, norms in by_target.items()
        },
        "base_gradient_tensors": 0,
        "lora_gradient_tensors": len(trainable),
    }


def update_parameters(optimizer, config, accelerator, optimizer_step, trainable):
    """Stage 2: warm up by one-based optimizer step, clip, update and measure."""
    lr = config.learning_rate * min(optimizer_step / max(1, config.warmup_steps), 1.0)
    for group in optimizer.param_groups:
        group["lr"] = lr
    before = {name: parameter.detach().clone() for name, parameter in trainable.items()}
    norm = torch.nn.utils.clip_grad_norm_(
        list(trainable.values()), config.max_grad_norm, error_if_nonfinite=True
    )
    clipped_norm = (
        torch.stack([p.grad.detach().float().norm() for p in trainable.values()])
        .norm()
        .item()
    )
    optimizer.step()
    if accelerator.optimizer_step_was_skipped:
        raise RuntimeError("Optimizer step was skipped; training aborted on overflow")
    deltas = {
        name: (parameter.detach() - before[name]).float().norm()
        for name, parameter in trainable.items()
    }
    values = torch.stack(list(deltas.values()))
    by_target = {target: [] for target in LORA_TARGET_MODULES}
    for name, delta in deltas.items():
        by_target[name.split(".lora_")[0].rsplit(".", 1)[-1]].append(delta)
    result = {
        "learning_rate": lr,
        "gradient_norm_before_clipping": norm.item(),
        "gradient_norm_after_clipping": clipped_norm,
        "updated": bool(torch.any(values > 0)),
        "changed_lora_tensors": int((values > 0).sum()),
        "lora_parameter_delta_norm": values.norm().item(),
        "lora_delta_norm_by_target": {
            name: torch.stack(norms).norm().item() for name, norms in by_target.items()
        },
        "optimizer_step_skipped": accelerator.optimizer_step_was_skipped,
    }
    optimizer.zero_grad(set_to_none=True)
    return result
