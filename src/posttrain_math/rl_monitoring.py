"""Durable RL observations and checkpoint completion markers; no reward shaping."""
from __future__ import annotations

import json
import math
import time
import uuid
from pathlib import Path

import torch
from transformers import TrainerCallback

from posttrain_math.answers import extract_final_boxed_numeric
from posttrain_math.artifacts import normalize_adapter_artifact
from posttrain_math.experiment_io import write_json
from posttrain_math.progress import ProgressReporter
from posttrain_math.rewards import completion_text, score_boxed_numeric_completion
from posttrain_math.rollout_records import (
    GenerationResult,
    diagnose_generation,
    trim_generated_ids,
)


def append_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
        handle.flush()


class RolloutAudit:
    def __init__(self, output: Path, *, group_size: int, eos_token_id: int, max_tokens: int):
        self.attempt = uuid.uuid4().hex
        self.root = output / 'attempts' / self.attempt
        self.group_size, self.eos_token_id, self.max_tokens = group_size, eos_token_id, max_tokens
        self.calls = self.completions = self.correct = self.numeric = self.groups = self.mixed = self.tokens = 0
        self.started = time.monotonic()
        self.reporter = ProgressReporter(output)

    def reward(self, completions, gt_numerator, gt_denominator, *, prompts, completion_ids,
               prompt_id, trainer_state, **kwargs):
        n = len(completions)
        if n % self.group_size or any(len(x) != n for x in
                                      (gt_numerator, gt_denominator, prompts, completion_ids, prompt_id)):
            raise ValueError('RL audit requires complete, aligned prompt groups')
        rewards = [score_boxed_numeric_completion(c, a, b)
                   for c, a, b in zip(completions, gt_numerator, gt_denominator, strict=True)]
        self.calls += 1
        numeric = mixed = tokens = 0
        for start in range(0, n, self.group_size):
            if len(set(prompt_id[start:start + self.group_size])) != 1:
                raise ValueError('RL rollout grouping changed; refusing misleading advantage diagnostics')
            count = sum(rewards[start:start + self.group_size])
            mixed += int(0 < count < self.group_size)
        for i, (completion, ids) in enumerate(zip(completions, completion_ids, strict=True)):
            text = completion_text(completion)
            ids, stop = trim_generated_ids(list(ids), eos_token_id=self.eos_token_id, max_new_tokens=self.max_tokens)
            valid = extract_final_boxed_numeric(text) is not None
            numeric += int(valid)
            tokens += len(ids)
            append_json(self.root / 'rollouts.jsonl', {
                'attempt_id': self.attempt, 'call': self.calls, 'policy_step': trainer_state.global_step,
                'group_index': i // self.group_size, 'sample_index': i % self.group_size,
                'prompt_id': prompt_id[i], 'prompt': prompts[i], 'generation': text,
                'generated_token_ids': ids, 'gt_numerator': int(gt_numerator[i]),
                'gt_denominator': int(gt_denominator[i]), 'reward': rewards[i], 'prediction_numeric': valid,
                'generation_diagnostics': diagnose_generation(GenerationResult(text, ids, stop)),
            })
        self.completions += n
        self.correct += int(sum(rewards))
        self.numeric += numeric
        self.groups += n // self.group_size
        self.mixed += mixed
        self.tokens += tokens
        stats = {'attempt_id': self.attempt, 'calls': self.calls, 'completions': self.completions,
                 'correct': self.correct, 'numeric': self.numeric, 'groups': self.groups,
                 'mixed_groups': self.mixed, 'zero_variance_groups': self.groups - self.mixed,
                 'generated_tokens': self.tokens, 'elapsed_seconds': time.monotonic() - self.started,
                 'scope': 'Generated rollouts in this attempt; interrupted updates may be replayed on resume.'}
        write_json(self.root / 'rollout_summary.json', stats)
        self.reporter.emit('rollouts_scored', step=trainer_state.global_step, batch_completions=n,
                           reward=sum(rewards) / n, numeric_rate=numeric / n,
                           mixed_groups=mixed, groups=n // self.group_size, generated_tokens=tokens,
                           gpu_peak_gib=torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0)
        return rewards


class RLProgressCallback(TrainerCallback):
    def __init__(self, output: Path, audit: RolloutAudit, base_source: dict, *, stop_after: int | None):
        self.output, self.audit, self.base_source, self.stop_after = output, audit, base_source, stop_after
        self.started = time.monotonic()
        self.initial_step = 0

    def on_train_begin(self, args, state, control, **kwargs):
        self.initial_step = state.global_step
        write_json(self.audit.root / 'attempt.json', {
            'attempt_id': self.audit.attempt, 'initial_step': state.global_step,
            'max_steps': state.max_steps, 'stop_after': self.stop_after, 'status': 'running',
        })
        self.audit.reporter.emit('rl_started', step=state.global_step, max_steps=state.max_steps)

    def on_step_begin(self, args, state, control, **kwargs):
        self.audit.reporter.emit('rl_update_started', step=state.global_step + 1, max_steps=state.max_steps)

    def on_step_end(self, args, state, control, **kwargs):
        done = state.global_step - self.initial_step
        self.audit.reporter.emit('rl_update_complete', step=state.global_step, max_steps=state.max_steps,
                                 eta_seconds=(time.monotonic() - self.started) / max(1, done)
                                 * max(0, state.max_steps - state.global_step))
        if self.stop_after is not None and state.global_step >= self.stop_after:
            control.should_training_stop = True
            control.should_save = True

    def on_log(self, args, state, control, logs=None, **kwargs):
        values = dict(logs or {})
        if any(isinstance(v, (float, int)) and not math.isfinite(v) for v in values.values()):
            raise FloatingPointError(f'Non-finite RL metrics at step {state.global_step}: {values}')
        append_json(self.audit.root / 'train_log.jsonl', {'step': state.global_step, **values})

        display = (
            ('loss', 'loss'),
            ('grad_norm', 'grad_norm'),
            ('learning_rate', 'lr'),
            ('reward', 'reward'),
            ('reward_std', 'reward_std'),
            ('frac_reward_zero_std', 'zero_std'),
            ('entropy', 'entropy'),
            ('completions/mean_length', 'mean_len'),
            ('num_tokens', 'tokens'),
            ('step_time', 'step_s'),
        )
        fields = []
        for key, label in display:
            if key not in values:
                continue
            value = values[key]
            rendered = f'{value:.6g}' if isinstance(value, float) else str(value)
            fields.append(f'{label}={rendered}')
        if fields:
            print(f"[rl-metrics] step={state.global_step} " + ' '.join(fields), flush=True)

    def on_save(self, args, state, control, **kwargs):
        path = self.output / f'checkpoint-{state.global_step}'
        if not normalize_adapter_artifact(path, self.base_source):
            raise RuntimeError(f'Missing adapter configuration in saved checkpoint: {path}')
        required = ('adapter_model.safetensors', 'optimizer.pt', 'scheduler.pt', 'trainer_state.json', 'rng_state.pth')
        if args.fp16:
            required += ('scaler.pt',)
        if not all((path / name).is_file() for name in required):
            raise RuntimeError(f'Incomplete RL checkpoint: {path}')
        write_json(path / 'complete.json', {'step': state.global_step, 'attempt_id': self.audit.attempt,
                                           'fp16': args.fp16})
        append_json(self.output / 'checkpoint_events.jsonl', {'step': state.global_step, 'path': path.name})
        self.audit.reporter.emit('rl_checkpoint_saved', step=state.global_step, path=str(path))

    def on_train_end(self, args, state, control, **kwargs):
        write_json(self.audit.root / 'attempt.json', {
            'attempt_id': self.audit.attempt, 'initial_step': self.initial_step,
            'final_step': state.global_step, 'max_steps': state.max_steps, 'status': 'stopped_cleanly',
        })
