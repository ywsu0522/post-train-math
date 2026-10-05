"""Exercise the actual pinned TRL RLOO trainer on a tiny CPU LoRA model."""
import json
from types import SimpleNamespace

import pandas as pd
import torch
from peft import LoraConfig, get_peft_model
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from posttrain_math.rl import train_rl
from posttrain_math.rl_pilot import complete_checkpoint


def test_real_rloo_smoke_checkpoint_resume_and_export(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    base = tmp_path / 'base'
    base.mkdir()
    (base / 'source.json').write_text(json.dumps({'repo_id': 'test/base', 'resolved_commit': 'test'}))
    vocab = {'<eos>': 0, '<unk>': 1, 'q': 2, r'\boxed{1}': 3, r'\boxed{2}': 4, 'work': 5}
    backend = Tokenizer(WordLevel(vocab, unk_token='<unk>'))
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token='<eos>', pad_token='<eos>',
                                        unk_token='<unk>', model_max_length=256)
    tokenizer.padding_side = 'left'

    def load(*a, **kw):
        torch.manual_seed(123)
        model = GPT2LMHeadModel(GPT2Config(vocab_size=6, n_positions=256, n_embd=8, n_layer=1, n_head=1,
                                          eos_token_id=0, pad_token_id=0, bos_token_id=2))
        model.config._name_or_path = str(base)
        model.config.save_pretrained(base)
        return get_peft_model(model, LoraConfig(r=2, lora_alpha=4, target_modules=['c_attn'],
                                               task_type='CAUSAL_LM')), tokenizer, base

    monkeypatch.setattr('posttrain_math.rl.load_sft_adapter', load)
    monkeypatch.setattr('posttrain_math.rl.resolve_rl_precision', lambda x: 'fp32')
    monkeypatch.setattr('posttrain_math.rl.get_distributed_context',
                        lambda: SimpleNamespace(world_size=1, is_main_process=True))
    data = tmp_path / 'data'
    data.mkdir()
    pd.DataFrame([{'problem': 'q', 'type': 'Algebra', 'level': 'Level 1', 'numeric_eligible': True,
                   'gt_numerator': 1, 'gt_denominator': 1} for _ in range(4)]).to_parquet(data / 'train.parquet')
    output = tmp_path / 'rl'
    config = {'algorithm': 'rloo', 'model_path': tmp_path, 'data_dir': data, 'prompt_name': 'boxed',
              'output_dir': output, 'max_steps': 3, 'learning_rate': 1e-6, 'per_device_batch_size': 1,
              'gradient_accumulation': None, 'global_batch_size': 8, 'num_generations': 4,
              'max_prompt_length': 128, 'max_completion_length': 4, 'temperature': 1.0, 'top_p': 1.0,
              'beta': 0.0, 'precision': 'fp32', 'gradient_checkpointing': True, 'logging_steps': 1,
              'save_steps': 1, 'save_total_limit': 2, 'seed': 83, 'limit_prompts': None, 'audit_rollouts': True}
    train_rl(**config, resume_from_checkpoint=None, stop_after_steps=1)
    checkpoint = complete_checkpoint(output)
    assert checkpoint.name == 'checkpoint-1'
    assert not (output / 'final-model').exists()
    smoke_weights = (checkpoint / 'adapter_model.safetensors').read_bytes()
    train_rl(**config, resume_from_checkpoint=checkpoint)
    assert complete_checkpoint(output).name == 'checkpoint-3'
    assert (output / 'final-model/base_model_source.json').exists()
    assert (output / 'final-model/tokenizer.json').exists()
    assert (output / 'train_summary.json').exists()
    assert (output / 'final-model/adapter_model.safetensors').read_bytes() != smoke_weights
    records = [json.loads(line) for p in (output / 'attempts').glob('*/rollouts.jsonl')
               for line in p.read_text().splitlines()]
    assert len(records) == 24
    assert {r['policy_step'] for r in records} == {0, 1, 2}
    assert 0 < sum(r['reward'] for r in records) < len(records)
