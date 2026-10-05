"""Record actual generation lengths and stopping conditions without changing outputs."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class GenerationResult:
    text: str
    token_ids: list[int]
    finish_reason: str


def trim_generated_ids(token_ids: list[int], *, eos_token_id: int, max_new_tokens: int):
    if eos_token_id in token_ids:
        return token_ids[:token_ids.index(eos_token_id) + 1], 'eos'
    return token_ids, 'length' if len(token_ids) >= max_new_tokens else 'other'


def diagnose_generation(result: GenerationResult) -> dict:
    return {'generated_tokens': len(result.token_ids), 'finish_reason': result.finish_reason,
            'hit_token_limit': result.finish_reason == 'length'}


def summarize_diagnostics(rows: list[dict]) -> dict:
    values = [r['generation_diagnostics'] for r in rows]
    n = len(values)
    return {'finish_reasons': dict(Counter(r['finish_reason'] for r in values)),
            'token_limit_rate': sum(r['hit_token_limit'] for r in values) / n if n else None,
            'mean_generated_tokens': sum(r['generated_tokens'] for r in values) / n if n else None}


class PilotRunner:
    """Single-prompt evaluation using exactly the RL loader and sampling defaults."""
    @classmethod
    def from_pretrained(cls, path, *, batch_size=1):
        import torch

        from posttrain_math.rl_common import load_sft_adapter, resolve_rl_precision

        if batch_size != 1:
            raise ValueError('Pilot evaluation requires batch size 1')
        instance = cls()
        instance.model, instance.tokenizer, _ = load_sft_adapter(
            path, precision=resolve_rl_precision('auto'), trainable=False)
        instance.model.to(torch.device('cuda'))
        instance.model.eval()
        return instance

    def iter_generate_records(self, prompts, **config):
        import torch

        for prompt in prompts:
            encoded = self.tokenizer(prompt, return_tensors='pt', add_special_tokens=False).to(self.model.device)
            with torch.inference_mode():
                output = self.model.generate(**encoded, **config)
            ids, reason = trim_generated_ids(output[0, encoded['input_ids'].shape[1]:].tolist(),
                                             eos_token_id=self.tokenizer.eos_token_id,
                                             max_new_tokens=config['max_new_tokens'])
            yield GenerationResult(self.tokenizer.decode(ids, skip_special_tokens=True), ids, reason)
