import json
from types import SimpleNamespace

import pandas as pd
import pytest

from posttrain_math.experiment_io import file_hash, read_json, write_json
from posttrain_math.rl_pilot import (
    complete_checkpoint,
    evaluate_worker,
    make_selection,
    report,
    train_worker,
)
from posttrain_math.rollout_records import GenerationResult


def frame(split, n=80):
    return pd.DataFrame([{'problem': f'{split} {i}', 'solution': 'SECRET_SOLUTION',
                          'level': 'Level 1' if i % 2 == 0 else 'Level 2', 'type': 'Algebra',
                          'numeric_eligible': True, 'gt_numerator': 1, 'gt_denominator': 1} for i in range(n)])


def test_selection_keeps_all_levels_without_solution_or_reward_filtering():
    train, dev = frame('train'), frame('dev')
    train.loc[2, 'problem'] = 'dev   0'
    train.loc[4, 'solution'] = 'LONG ' * 3000
    train.loc[6, 'numeric_eligible'] = False
    selected = make_selection(train, dev, token_length=lambda s: 20, seed=83)
    chosen = selected['train']
    assert len(chosen) == 78
    assert {c['level'] for c in chosen} == {'Level 1', 'Level 2'}
    assert all('SECRET' not in c['prompt'] and 'LONG' not in c['prompt'] for c in chosen)
    assert 'train 4' in [c['problem'] for c in chosen]
    assert 'dev   0' not in [c['problem'] for c in chosen]
    groups = [{c['problem_sha256'] for c in selected[key]} for key in ('train', 'dev', 'sampled_dev')]
    assert len(groups[1]) == 64 and len(groups[2]) == 16
    assert not groups[0] & groups[1] and groups[2] <= groups[1]
    assert selected == make_selection(train, dev, token_length=lambda s: 20, seed=83)


def test_eval_pairing_is_fixed_and_resume_does_not_reload_gpu(tmp_path, monkeypatch):
    from posttrain_math.rollout_records import PilotRunner

    selected = make_selection(frame('train'), frame('dev'), token_length=lambda s: 20, seed=83,
                              dev_size=4, sample_size=2)
    write_json(tmp_path / 'selection.json', selected)
    adapter = tmp_path / 'adapter'
    adapter.mkdir()
    (adapter / 'adapter_model.safetensors').write_bytes(b'sft')
    final = tmp_path / 'rl/final-model'
    final.mkdir(parents=True)
    (final / 'adapter_model.safetensors').write_bytes(b'rl')
    write_json(tmp_path / 'rl/complete.json', {'weights_sha256': file_hash(final / 'adapter_model.safetensors')})
    loads = []

    class Runner:
        def iter_generate_records(self, prompts, **kwargs):
            yield GenerationResult(r'\boxed{1}', [1, 0], 'eos')

    def load(*a, **kw):
        loads.append(a)
        return Runner()

    monkeypatch.setattr(PilotRunner, 'from_pretrained', load)
    args = SimpleNamespace(output_dir=tmp_path, adapter=adapter, seed=83, max_new_tokens=32)
    for stage in ('baseline', 'evaluate', 'evaluate'):
        args.stage = stage
        evaluate_worker(args)
    assert len(loads) == 2
    result = report(tmp_path)
    assert result['paired']['sampled_dev_sampled']['correct'] == {'gained': 0, 'lost': 0}
    assert result['evaluations']['sft']['sampled_dev_sampled']['all_correct_groups'] == 2


def checkpoint(root, step):
    p = root / f'checkpoint-{step}'
    p.mkdir(parents=True)
    for name in ('adapter_model.safetensors', 'optimizer.pt', 'scheduler.pt', 'rng_state.pth'):
        (p / name).write_bytes(b'data')
    for name in ('adapter_config.json', 'base_model_source.json'):
        write_json(p / name, {})
    write_json(p / 'trainer_state.json', {'global_step': step})
    return p


def test_checkpoint_marker_and_final_export_recovery_avoid_extra_update(tmp_path, monkeypatch):
    output = tmp_path / 'rl'
    a = checkpoint(output, 2)
    b = checkpoint(output, 50)
    assert complete_checkpoint(output) is None
    write_json(a / 'complete.json', {'step': 2})
    assert complete_checkpoint(output) == a
    write_json(b / 'complete.json', {'step': 50, 'fp16': True})
    assert complete_checkpoint(output) == a
    (b / 'scaler.pt').write_bytes(b'scaler')
    monkeypatch.setattr('posttrain_math.rl.train_rl', lambda **kw: pytest.fail('Must not retrain at max steps'))
    args = SimpleNamespace(stage='train', output_dir=tmp_path, adapter=tmp_path / 'sft', max_steps=50)
    train_worker(args)
    assert read_json(output / 'complete.json')['max_steps'] == 50
    assert (output / 'final-model/adapter_model.safetensors').read_bytes() == b'data'


def test_rollout_audit_preserves_binary_reward_and_counts_groups(tmp_path):
    from posttrain_math.rl_monitoring import RolloutAudit

    audit = RolloutAudit(tmp_path, group_size=4, eos_token_id=0, max_tokens=16)
    rewards = audit.reward([r'\boxed{1}', r'\boxed{2}', 'no box', r'\boxed{1}'] * 2,
                           [1] * 8, [1] * 8, prompts=['Q'] * 8, completion_ids=[[1, 0]] * 8,
                           prompt_id=['a'] * 4 + ['b'] * 4, trainer_state=SimpleNamespace(global_step=0))
    assert rewards == [1, 0, 0, 1] * 2
    summary = read_json(audit.root / 'rollout_summary.json')
    assert summary['mixed_groups'] == summary['groups'] == 2
    assert summary['correct'] == 4 and summary['numeric'] == 6
    rows = [json.loads(s) for s in (audit.root / 'rollouts.jsonl').read_text().splitlines()]
    assert len(rows) == 8 and rows[0]['generated_token_ids'] == [1, 0]


def test_prepare_pins_selection_weights_data_and_options(tmp_path, monkeypatch):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel

    from posttrain_math.rl_pilot import prepare

    monkeypatch.chdir(tmp_path)
    adapter, data = tmp_path / 'adapter', tmp_path / 'data'
    adapter.mkdir()
    data.mkdir()
    (adapter / 'adapter_model.safetensors').write_bytes(b'new numeric-cohort SFT weights')
    write_json(adapter / 'adapter_config.json', {'peft_type': 'LORA'})
    write_json(adapter / 'tokenizer_config.json', {})
    source = {'repo_id': 'test/base', 'resolved_commit': 'pinned'}
    write_json(adapter / 'base_model_source.json', source)
    Tokenizer(WordLevel({'<unk>': 0}, unk_token='<unk>')).save(str(adapter / 'tokenizer.json'))
    for split in ('train', 'dev'):
        frame(split).to_parquet(data / f'{split}.parquet', index=False)
    base = tmp_path / 'models/olmo-2-0425-1b'
    write_json(base / 'source.json', source)
    (base / 'tokenizer.json').write_bytes((adapter / 'tokenizer.json').read_bytes())
    for name in ('pyproject.toml', 'uv.lock'):
        (tmp_path / name).write_text('')
    args = SimpleNamespace(adapter=adapter, data_dir=data, output_dir=tmp_path / 'run', seed=83,
                           max_steps=50, learning_rate=1e-6, global_batch_size=8, max_new_tokens=512)
    prepare(args)
    pinned_plan = read_json(args.output_dir / 'plan.json')
    assert pinned_plan['adapter']['adapter_model.safetensors'] == file_hash(adapter / 'adapter_model.safetensors')
    original = (args.output_dir / 'selection.json').read_bytes()
    prepare(args)
    assert (args.output_dir / 'selection.json').read_bytes() == original
    assert len(pd.read_parquet(args.output_dir / 'rl-data/train.parquet')) == 80
    args.max_steps = 60
    with pytest.raises(ValueError, match='Inputs/code/options changed'):
        prepare(args)
    args.max_steps = 50
    changed = frame('train')
    changed.loc[0, 'problem'] = 'changed problem'
    changed.to_parquet(data / 'train.parquet', index=False)
    with pytest.raises(ValueError, match='Inputs/code/options changed'):
        prepare(args)
    frame('train').to_parquet(data / 'train.parquet', index=False)
    (adapter / 'adapter_model.safetensors').write_bytes(b'different weights')
    with pytest.raises(ValueError, match='Inputs/code/options changed'):
        prepare(args)
    assert read_json(args.output_dir / 'plan.json') == pinned_plan
    args.output_dir = tmp_path / 'new-run'
    prepare(args)
    assert read_json(args.output_dir / 'plan.json')['adapter']['adapter_model.safetensors'] == file_hash(
        adapter / 'adapter_model.safetensors')
