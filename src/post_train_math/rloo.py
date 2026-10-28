"""An observable RLOO iteration: generate -> reward/advantage/backward -> AdamW.

Only LoRA matrices receive gradients. The frozen reference is this same PEFT
model with adapters disabled. No Trainer subclass or hidden training loop.
"""

from __future__ import annotations

import json
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence

from post_train_math.answers import numeric_reward
from post_train_math.checkpoints import load_checkpoint, save_checkpoint
from post_train_math.training import LORA_TARGET_MODULES, validate_batch


@dataclass
class Rollouts:
    records: list[dict]
    prompt_ids: list[list[int]]
    completion_ids: list[list[int]]
    texts: list[str]
    sampled_indices: list[int]


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
    resume=None,
    reward_fn=numeric_reward,
):
    """One visible loop. Each iteration consumes a NEW rollout batch, updates once."""
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
    step, history = (
        (0, [])
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
        lora_targets=list(LORA_TARGET_MODULES),
        lora_parameter_tensors=len(trainable),
        trainable_parameters=sum(p.numel() for p in trainable.values()),
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

        # Stage 1.2: verifier + frozen reference -> LOO advantages -> LoRA .grad.
        log_stage(accelerator, config.output_dir, iteration, "1.2", status="start")
        gradient_stats = estimate_gradients(
            model,
            tokenizer,
            rollouts,
            optimizer,
            config,
            accelerator,
            iteration,
            trainable,
            reward_fn,
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
            optimizer, config, accelerator, iteration, trainable
        )
        log_stage(
            accelerator,
            config.output_dir,
            iteration,
            "2",
            status="complete",
            **update_stats,
        )
        history.append({"step": iteration, **gradient_stats, **update_stats})

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


def sequence_log_probs(model, tokenizer, rollouts, start, end, device):
    """Differentiable sum of completion-token log probabilities, including EOS."""
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
    return (token_logps * answer_mask).sum(1)


def leave_one_out_advantages(rewards, group_size):
    if group_size < 2 or rewards.numel() % group_size:
        raise ValueError("RLOO needs complete groups with G >= 2")
    grouped = rewards.reshape(-1, group_size)
    baseline = (grouped.sum(1, keepdim=True) - grouped) / (group_size - 1)
    return (grouped - baseline).reshape(-1).detach()


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


def estimate_gradients(
    model,
    tokenizer,
    rollouts,
    optimizer,
    config,
    accelerator,
    iteration,
    trainable,
    reward_fn=numeric_reward,
):
    """Stage 1.2: detached rewards/reference/advantages, then LoRA backward.

    A no-grad policy pass supplies the sampled KL reward. A fresh grad-enabled
    pass accumulates the on-policy REINFORCE estimator into each LoRA .grad.
    Microbatch sums are divided by the local rollout count; DDP averages ranks.
    """
    policy = accelerator.unwrap_model(model)
    count = len(rollouts.records)
    micro = config.per_device_train_batch_size
    raw_rewards = torch.tensor(
        reward_fn(
            completions=rollouts.texts,
            gt_numerator=[row["gt_numerator"] for row in rollouts.records],
            gt_denominator=[row["gt_denominator"] for row in rollouts.records],
        ),
        device=accelerator.device,
        dtype=torch.float32,
    )
    if raw_rewards.shape != (count,) or not torch.isfinite(raw_rewards).all():
        raise ValueError("Verifier must return one finite reward per completion")
    policy.eval()
    old_logps, ref_logps = [], []
    for start in range(0, count, micro):
        with torch.no_grad(), accelerator.autocast():
            old_logps.append(
                sequence_log_probs(
                    policy,
                    tokenizer,
                    rollouts,
                    start,
                    start + micro,
                    accelerator.device,
                )
            )
            with policy.disable_adapter():
                ref_logps.append(
                    sequence_log_probs(
                        policy,
                        tokenizer,
                        rollouts,
                        start,
                        start + micro,
                        accelerator.device,
                    )
                )
    scores = accelerator.gather(
        torch.stack([raw_rewards, torch.cat(old_logps), torch.cat(ref_logps)], dim=1)
    )
    kl = scores[:, 1] - scores[:, 2]
    rewards = scores[:, 0] - config.beta * kl
    advantages = leave_one_out_advantages(rewards, config.num_generations)
    log_stage(
        accelerator,
        config.output_dir,
        iteration,
        "1.2",
        status="advantages",
        verifier_rewards=scores[:, 0].tolist(),
        sequence_kl=kl.tolist(),
        policy_log_probs_for_kl=scores[:, 1].tolist(),
        reference_log_probs=scores[:, 2].tolist(),
        shaped_rewards=rewards.tolist(),
        advantages=advantages.reshape(-1, config.num_generations).tolist(),
        advantage_min=advantages.min().item(),
        advantage_max=advantages.max().item(),
        advantage_mean=advantages.mean().item(),
        advantage_std=advantages.std(unbiased=False).item(),
    )
    local = advantages[
        accelerator.process_index * count : (accelerator.process_index + 1) * count
    ]
    optimizer.zero_grad(set_to_none=True)
    model.train()
    loss_value = 0.0
    gradient_logps = []
    for start in range(0, count, micro):
        sync = nullcontext() if start + micro >= count else accelerator.no_sync(model)
        with sync:
            with accelerator.autocast():
                logps = sequence_log_probs(
                    model, tokenizer, rollouts, start, start + micro, accelerator.device
                )
                if not logps.requires_grad:
                    raise RuntimeError(
                        "Policy log probabilities must retain the LoRA graph"
                    )
                gradient_logps.append(logps.detach())
                # L = -1/(BG) sum advantage * sum_h log pi(y_h | prompt, y_<h).
                loss = -(local[start : start + micro] * logps).sum() / count
            accelerator.backward(loss)
        loss_value += loss.detach().item()
    accelerator.unscale_gradients(optimizer)
    stats = gradient_statistics(model, trainable)
    stats["loss"] = accelerator.reduce(
        torch.tensor(loss_value, device=accelerator.device), reduction="mean"
    ).item()
    stats["reward_mean"] = scores[:, 0].mean().item()
    stats["policy_log_probs"] = accelerator.gather(torch.cat(gradient_logps)).tolist()
    stats["policy_log_probs_require_grad"] = logps.requires_grad
    stats["reference_log_probs_require_grad"] = any(p.requires_grad for p in ref_logps)
    return stats


def update_parameters(optimizer, config, accelerator, iteration, trainable):
    """Stage 2: set LR, clip accumulated gradients, step AdamW, measure changes."""
    lr = config.learning_rate * min(iteration / max(1, config.warmup_steps), 1.0)
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
