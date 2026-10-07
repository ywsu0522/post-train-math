"""Resumable, single-T4 RLOO pilot from a numeric-cohort SFT adapter."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
import zipfile
from pathlib import Path

from posttrain_math.experiment_io import (
    code_hash,
    file_hash,
    package_results,
    read_json,
    write_json,
)
from posttrain_math.progress import stream_command
from posttrain_math.rl_sampling import (
    coverage,
    fingerprint,
    run_job,
    summarize,
    tasks,
)

CONTRACT = 'numeric-rloo-sft2ep-v1'
OPTIONS = ('seed', 'max_steps', 'learning_rate', 'global_batch_size', 'max_new_tokens', 'precision', 'num_generations', 'levels')


def problem_hash(problem: str) -> str:
    return hashlib.sha256(' '.join(problem.split()).encode()).hexdigest()


def make_selection(train, dev, *, token_length, seed: int, levels=None, dev_size: int = 64, sample_size: int = 16, support_per_level: int = 16) -> dict:
    """Apply one level scope to train/validation; keep support scan inside training data."""
    from posttrain_math.prompting import get_prompt_formatter

    required = {'problem', 'level', 'type', 'numeric_eligible', 'gt_numerator', 'gt_denominator'}
    if any(not required <= set(frame.columns) for frame in (train, dev)):
        raise ValueError('Outdated processed data; run posttrain-math data prepare')
    formatter = get_prompt_formatter('boxed')

    def cases(frame, split):
        result, seen = [], set()
        for index, row in frame.reset_index(drop=True).iterrows():
            key = problem_hash(str(row['problem']))
            if not row['numeric_eligible'] or key in seen:
                continue
            seen.add(key)
            prompt = formatter(str(row['problem']))
            if token_length(prompt) > 1024:
                continue
            case = {'split': split, 'source_index': int(index), 'problem': str(row['problem']),
                    'problem_sha256': key, 'prompt': prompt, 'gt_numerator': int(row['gt_numerator']),
                    'gt_denominator': int(row['gt_denominator']), 'subject': str(row['type']),
                    'level': str(row['level'])}
            result.append({**case, 'question_id': fingerprint(case)})
        return result

    level_values = None if levels is None else sorted(set(levels))
    if level_values is not None and any(level not in range(1, 6) for level in level_values):
        raise ValueError('levels must be drawn from 1..5')
    allowed_levels = None if level_values is None else {f'Level {level}' for level in level_values}

    dev_keys = {problem_hash(str(p)) for p in dev['problem']}
    train_cases = [c for c in cases(train, 'train') if c['problem_sha256'] not in dev_keys]
    dev_cases = cases(dev, 'dev')
    if allowed_levels is not None:
        train_cases = [c for c in train_cases if c['level'] in allowed_levels]
        dev_cases = [c for c in dev_cases if c['level'] in allowed_levels]
    if len(train_cases) < 8:
        raise ValueError('Too few numeric training prompts after level filtering')

    random.Random(seed).shuffle(train_cases)
    random.Random(seed + 1).shuffle(dev_cases)
    chosen = dev_cases[:dev_size]
    if len(chosen) != dev_size or not 0 < sample_size <= dev_size:
        raise ValueError('Insufficient level-filtered dev rows for the fixed validation')

    support_train = []
    support_levels = sorted({c['level'] for c in train_cases})
    for label in support_levels:
        candidates = [c for c in train_cases if c['level'] == label]
        if len(candidates) < support_per_level:
            raise ValueError(f'Insufficient training rows for support scan: {label}')
        support_train.extend(candidates[:support_per_level])

    if level_values is None:
        basis = 'Numeric-eligible train and validation pools; all levels/types.'
    else:
        labels = ', '.join(f'Level {level}' for level in level_values)
        basis = f'Numeric-eligible train and validation pools restricted to {labels}.'
    return {'train': train_cases, 'support_train': support_train,
            'dev': chosen, 'sampled_dev': chosen[:sample_size], 'levels': level_values,
            'selection_basis': basis,
            'support_scope': 'Fixed training-set subset; used only to estimate reward support and choose K/G.',
            'validation_scope': 'Held-out dev split within the same level scope; never used for gradients or K/G selection.'}


def validate_initial_adapter(adapter: Path) -> None:
    from tokenizers import Tokenizer

    base = Path("models/olmo-2-0425-1b")

    required = (
        "adapter_model.safetensors",
        "adapter_config.json",
        "base_model_source.json",
        "tokenizer.json",
        "tokenizer_config.json",
    )
    missing = [
        name
        for name in required
        if not (adapter / name).is_file()
    ]
    if missing:
        raise ValueError(
            f"SFT adapter is incomplete; missing: {missing}"
        )

    config = read_json(adapter / "adapter_config.json")
    if config.get("peft_type") != "LORA":
        raise ValueError(
            "RLOO pilot requires a LoRA SFT adapter"
        )

    expected = read_json(
        adapter / "base_model_source.json"
    )
    actual = read_json(
        base / "source.json"
    )
    if any(
        not expected.get(key)
        or expected[key] != actual.get(key)
        for key in ("repo_id", "resolved_commit")
    ):
        raise ValueError(
            "Adapter/base provenance mismatch"
        )

    def tokenizer_contract(path):
        value = json.loads(
            Tokenizer.from_file(str(path)).to_str()
        )
        value.pop("padding", None)
        value.pop("truncation", None)

        identity = {
            "type": "TemplateProcessing",
            "single": [
                {
                    "Sequence": {
                        "id": "A",
                        "type_id": 0,
                    }
                }
            ],
            "pair": [
                {
                    "Sequence": {
                        "id": "A",
                        "type_id": 0,
                    }
                },
                {
                    "Sequence": {
                        "id": "B",
                        "type_id": 1,
                    }
                },
            ],
            "special_tokens": {},
        }

        if value.get("post_processor") == identity:
            value["post_processor"] = None

        return value

    if (
        tokenizer_contract(
            adapter / "tokenizer.json"
        )
        != tokenizer_contract(
            base / "tokenizer.json"
        )
    ):
        raise ValueError(
            "Adapter tokenizer differs from "
            "the pinned base tokenizer"
        )


def prepare(args):
    import pandas as pd
    from tokenizers import Tokenizer

    root = args.output_dir
    print('Checking SFT adapter, base provenance and tokenizer...', flush=True)
    validate_initial_adapter(args.adapter)
    contract = {'contract': CONTRACT, 'algorithm': 'rloo', 'reward': 'boxed-numeric-v1 binary correctness only',
                'options': {name: getattr(args, name) for name in OPTIONS},
                'sampling': {'temperature': 1.0, 'top_p': 1.0, 'top_k': 0,
                             'num_generations': args.num_generations, 'beta': 0.0},
                'adapter': {n: file_hash(args.adapter / n) for n in
                            ('adapter_model.safetensors', 'adapter_config.json', 'tokenizer.json',
                             'tokenizer_config.json', 'base_model_source.json')},
                'data': {s: file_hash(args.data_dir / f'{s}.parquet') for s in ('train', 'dev')},
                'base_source': file_hash(Path('models/olmo-2-0425-1b/source.json')),
                'code_sha256': code_hash(), 'code_base': '4339742'}
    plan = root / 'plan.json'
    if plan.exists() and read_json(plan) != contract:
        raise ValueError('Inputs/code/options changed; use a new output directory')
    tokenizer = Tokenizer.from_file(str(args.adapter / 'tokenizer.json'))
    train, dev = (pd.read_parquet(args.data_dir / f'{s}.parquet') for s in ('train', 'dev'))
    selected = make_selection(train, dev,
                              token_length=lambda s: len(tokenizer.encode(s, add_special_tokens=False).ids),
                              seed=args.seed, levels=args.levels)
    if (root / 'selection.json').exists() and read_json(root / 'selection.json') != selected:
        raise ValueError('Saved dataset selection changed')
    root.mkdir(parents=True, exist_ok=True)
    data_path = root / 'rl-data/train.parquet'
    data_path.parent.mkdir(exist_ok=True)
    chosen = train.iloc[[c['source_index'] for c in selected['train']]].copy()
    if data_path.exists():
        if not pd.read_parquet(data_path).equals(chosen.reset_index(drop=True)):
            raise ValueError('Prepared RL dataset changed')
    else:
        chosen.reset_index(drop=True).to_parquet(data_path, index=False)
    write_json(plan, contract)
    write_json(root / 'selection.json', selected)
    level_counts = {str(level): int(count) for level, count in chosen['level'].value_counts().sort_index().items()}
    write_json(root / 'dataset_manifest.json', {'train_parquet_sha256': file_hash(data_path),
               'train_prompts': len(chosen), 'levels': args.levels,
               'train_level_counts': level_counts, 'support_prompts': len(selected['support_train']),
               'support_samples_per_prompt': 32, 'num_generations': args.num_generations,
               'validation_greedy': 64, 'validation_sampled': 16,
               'solution_usage': 'Gold numeric extraction only; no length selection; model sees problem prompt only.'})
    if not (root / 'code_snapshot.zip').exists():
        with zipfile.ZipFile(root / 'code_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as z:
            for p in sorted(Path('src/posttrain_math').glob('*.py')) + [Path('pyproject.toml'), Path('uv.lock')]:
                z.write(p, p.as_posix())
    print(f"Prepared {len(chosen)} train prompts; baseline/final each: 64 greedy + 64 sampled answers.", flush=True)
    print(f"RL: {args.max_steps} total updates, "
          f"G={args.num_generations}, {args.global_batch_size // args.num_generations} prompt groups/update, "
          f"{args.max_steps * args.global_batch_size} planned rollouts; no format-rate gate.", flush=True)


def complete_checkpoint(output: Path) -> Path | None:
    candidates = []
    for p in output.glob('checkpoint-*'):
        if not p.name.rsplit('-', 1)[-1].isdigit():
            continue
        if all((p / n).is_file() for n in ('complete.json', 'adapter_model.safetensors', 'adapter_config.json',
                                          'base_model_source.json', 'optimizer.pt', 'scheduler.pt',
                                          'trainer_state.json', 'rng_state.pth')):
            step = int(p.name.rsplit('-', 1)[-1])
            marker = read_json(p / 'complete.json')
            if marker.get('fp16') and not (p / 'scaler.pt').is_file():
                continue
            if marker['step'] == read_json(p / 'trainer_state.json')['global_step'] == step:
                candidates.append(p)
    return max(candidates, key=lambda p: int(p.name.rsplit('-', 1)[-1]), default=None)


def train_worker(args):
    from posttrain_math.rl import train_rl

    root, output = args.output_dir, args.output_dir / 'rl'
    done = output / 'complete.json'
    if done.exists():
        if read_json(done)['weights_sha256'] != file_hash(output / 'final-model/adapter_model.safetensors'):
            raise ValueError('Final RL weights changed')
        print('RL already complete.', flush=True)
        return
    checkpoint = complete_checkpoint(output)
    step = int(checkpoint.name.rsplit('-', 1)[-1]) if checkpoint else 0
    if step >= args.max_steps:
        # Training reached its budget before a session died during final export.
        # Recover the adapter without accidentally performing an extra update.
        final = output / 'final-model'
        final.mkdir(parents=True, exist_ok=True)
        for name in ('adapter_model.safetensors', 'adapter_config.json', 'base_model_source.json',
                     'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json'):
            source = checkpoint / name
            if not source.exists():
                source = args.adapter / name
            if source.exists():
                shutil.copy2(source, final / name)
        summary = {'recovered_from_checkpoint': checkpoint.name, 'global_step': step,
                   'log_history': read_json(checkpoint / 'trainer_state.json').get('log_history', [])}
        write_json(output / 'train_summary.json', summary)
        write_json(done, {'max_steps': args.max_steps, 'weights_sha256': file_hash(
            final / 'adapter_model.safetensors'), 'train_summary_sha256': fingerprint(summary)})
        print('Recovered final model from the completed training checkpoint.', flush=True)
        return
    if args.stage == 'smoke' and step >= 2:
        print(f'Smoke already complete; latest checkpoint step={step}', flush=True)
        return
    validate_initial_adapter(args.adapter)
    print(f'Loading RL model; resuming step {step}; checkpoints and logs persist in the output directory.', flush=True)
    train_rl(algorithm='rloo', model_path=args.adapter, data_dir=root / 'rl-data', prompt_name='boxed',
             output_dir=output, max_steps=args.max_steps, learning_rate=args.learning_rate,
             per_device_batch_size=1, gradient_accumulation=None, global_batch_size=args.global_batch_size,
             num_generations=args.num_generations, max_prompt_length=1024,
             max_completion_length=args.max_new_tokens,
             temperature=1.0, top_p=1.0, beta=0.0, precision=args.precision, gradient_checkpointing=True,
             logging_steps=1, save_steps=10, save_total_limit=2, seed=args.seed, limit_prompts=None,
             resume_from_checkpoint=checkpoint, audit_rollouts=True,
             stop_after_steps=2 if args.stage == 'smoke' else None)
    if args.stage == 'train':
        summary = read_json(output / 'train_summary.json')
        write_json(done, {'max_steps': args.max_steps, 'weights_sha256': file_hash(
            output / 'final-model/adapter_model.safetensors'), 'train_summary_sha256': fingerprint(summary)})




def support_worker(args):
    """Estimate SFT reward support on a fixed subset of training prompts."""
    from posttrain_math.rollout_records import PilotRunner

    root = args.output_dir
    selected = read_json(root / 'selection.json')
    adapter = args.adapter
    runner = None

    def get_runner():
        nonlocal runner
        if runner is None:
            print('Loading SFT adapter for training-set reward-support scan...', flush=True)
            runner = PilotRunner.from_pretrained(adapter, batch_size=1)
        return runner

    config = {'max_new_tokens': args.max_new_tokens, 'do_sample': True,
              'temperature': 1.0, 'top_p': 1.0, 'top_k': 0}
    expected = tasks(selected['support_train'], 32, args.seed + 200)
    rows = run_job(root / 'support' / 'sft_k32' / 'predictions.jsonl',
                   expected, config, get_runner)

    ks = (4, 8, 16, 32)
    curves = {}
    for k in ks:
        subset = [row for row in rows if row['sample_index'] < k]
        curves[str(k)] = coverage(subset, k)

    per_level = {}
    for label in sorted({case['level'] for case in selected['support_train']}):
        level_rows = [row for row in rows if row['level'] == label]
        per_level[label] = {}
        for k in ks:
            subset = [row for row in level_rows if row['sample_index'] < k]
            per_level[label][str(k)] = coverage(subset, k)

    viable = [k for k in ks if curves[str(k)]['mixed_group_rate'] >= 0.4]
    recommendation = min(viable) if viable else None
    summary = {
        'source': 'training split only',
        'questions': len(selected['support_train']),
        'samples_per_question': 32,
        'ks': list(ks),
        'curve': curves,
        'per_level': per_level,
        'recommended_k': recommendation,
        'criterion': 'smallest K with observed mixed_group_rate >= 0.40',
        'validation_used': False,
    }
    write_json(root / 'support' / 'summary.json', summary)
    print('Reward-support scan complete.', flush=True)
    for k in ks:
        stats = curves[str(k)]
        print(f"K={k:>2} pass@K={stats['observed_success_at_k']:.1%} "
              f"mixed@K={stats['mixed_group_rate']:.1%} "
              f"zero_var={stats['zero_reward_variance_group_rate']:.1%}", flush=True)
    print(f"recommended_k={recommendation}", flush=True)


def evaluate_worker(args):
    from posttrain_math.rollout_records import PilotRunner

    root = args.output_dir
    name = 'sft' if args.stage == 'baseline' else 'rloo'
    adapter = args.adapter if name == 'sft' else root / 'rl/final-model'
    if name == 'rloo' and not (root / 'rl/complete.json').exists():
        raise ValueError('Complete RL training before final evaluation')
    selected = read_json(root / 'selection.json')
    model_hash = file_hash(adapter / 'adapter_model.safetensors')
    if name == 'rloo' and read_json(root / 'rl/complete.json')['weights_sha256'] != model_hash:
        raise ValueError('Final RL weights changed after training completed')
    provenance = root / 'evals' / name / 'model.json'
    if provenance.exists() and read_json(provenance)['weights_sha256'] != model_hash:
        raise ValueError('Evaluation model changed')
    write_json(provenance, {'weights_sha256': model_hash})
    runner = None

    def get_runner():
        nonlocal runner
        if runner is None:
            print(f'Loading {name} adapter for fixed dev evaluation...', flush=True)
            runner = PilotRunner.from_pretrained(adapter, batch_size=1)
        return runner

    blocks = {}
    for cohort, sample in (('dev', False), ('sampled_dev', True)):
        key = cohort + ('_sampled' if sample else '_greedy')
        config = {'max_new_tokens': args.max_new_tokens, 'do_sample': sample}
        if sample:
            config.update(temperature=1.0, top_p=1.0, top_k=0)
        rows = run_job(root / 'evals' / name / key / 'predictions.jsonl',
                       tasks(selected[cohort], 4 if sample else 1, args.seed + 100), config, get_runner)
        blocks[key] = coverage(rows, 4) if sample else summarize(rows)
        if sample:
            blocks[key]['scope'] = 'Fixed dev diagnostic; no training or checkpoint selection on these answers.'
    write_json(root / 'evals' / name / 'summary.json', blocks)


def report(root: Path) -> dict:
    result = {'evaluations': {}, 'training_complete': (root / 'rl/complete.json').exists()}
    support = root / 'support/summary.json'
    if support.exists():
        result['support_scan'] = read_json(support)
    lines = ['# RLOO pilot', '', 'Correctness-only RL; fixed 50-step budget by default. No box-rate gate.', '',
             '| Model | Dev cohort | N | Numeric box | Correct | Token limit |',
             '| --- | --- | ---: | ---: | ---: | ---: |']
    for name in ('sft', 'rloo'):
        p = root / 'evals' / name / 'summary.json'
        if not p.exists():
            continue
        result['evaluations'][name] = read_json(p)
        for key, stats in read_json(p).items():
            lines.append(f"| {name} | {key} | {stats['n']} | {stats['numeric_rate']:.2%} | "
                         f"{stats['accuracy']:.2%} | {stats['diagnostics']['token_limit_rate']:.2%} |")
    if set(result['evaluations']) == {'sft', 'rloo'}:
        result['paired'] = {}
        for key in result['evaluations']['sft']:
            a, b = ([json.loads(s) for s in (root / 'evals' / model / key / 'predictions.jsonl')
                     .read_text(encoding='utf-8').splitlines()] for model in ('sft', 'rloo'))
            if [r['task_id'] for r in a] != [r['task_id'] for r in b]:
                raise ValueError('Evaluation pairing mismatch')
            result['paired'][key] = {field: {
                'gained': sum(not x[field] and y[field] for x, y in zip(a, b, strict=True)),
                'lost': sum(x[field] and not y[field] for x, y in zip(a, b, strict=True)),
            } for field in ('correct', 'prediction_numeric')}
        lines += ['', 'Paired correct/numeric gains and losses are in report.json.']
    attempts = []
    for p in sorted((root / 'rl/attempts').glob('*/rollout_summary.json')):
        stat = read_json(p)
        attempts.append(stat)
        lines += ['', (f"Attempt {p.parent.name}: {stat['completions']} completions; "
                  f"reward={stat['correct'] / stat['completions']:.2%}; "
                  f"mixed={stat['mixed_groups']}/{stat['groups']}; generated tokens={stat['generated_tokens']}.")]
    result['rollout_attempts'] = attempts
    lines += ['', 'Attempts include compute spent on interrupted/replayed updates; do not sum them as unique updates.',
              'Use entropy/grad-norm/train logs and checkpoint events to inspect optimization stability.',
              'The sampled 16 questions are a subset of the 64 greedy dev questions; report these metrics separately.',
              'This small pilot measures feasibility and directional change, not statistical proof or reasoning validity.',
              'Training reward gains alone do not establish generalization. No extra SFT/RL is launched by the report.']
    write_json(root / 'report.json', result)
    (root / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'support', 'baseline', 'smoke', 'train', 'evaluate', 'report', 'package'))
    parser.add_argument('--adapter', type=Path)
    parser.add_argument('--data-dir', type=Path, default=Path('data/processed'))
    parser.add_argument('--output-dir', type=Path, default=Path('runs') / CONTRACT)
    parser.add_argument('--max-steps', type=int, default=50)
    parser.add_argument('--learning-rate', type=float, default=1e-6)
    parser.add_argument('--global-batch-size', type=int, default=8)
    parser.add_argument('--max-new-tokens', type=int, default=512)
    parser.add_argument('--precision', choices=('auto', 'fp16', 'bf16', 'fp32'), default='auto')
    parser.add_argument('--num-generations', type=int, default=4)
    parser.add_argument('--levels', type=int, nargs='+', choices=range(1, 6), default=None)
    parser.add_argument('--seed', type=int, default=83)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.levels is not None:
        args.levels = sorted(set(args.levels))
    root = args.output_dir
    if args.stage in ('report', 'package'):
        if not root.is_dir():
            parser.error('Output directory does not exist')
        report(root)
        if args.stage == 'package':
            print(package_results(root))
        return
    if args.adapter is None:
        parser.error('--adapter must point to the new SFT final-model')
    if (args.max_steps < 3 or args.num_generations < 2
            or args.global_batch_size < args.num_generations
            or args.global_batch_size % args.num_generations
            or args.max_new_tokens <= 0 or not 0 < args.learning_rate < float('inf')):
        parser.error('Invalid RL budget, group size, global batch or learning rate')
    if args.worker:
        if args.stage in ('smoke', 'train'):
            train_worker(args)
        elif args.stage == 'support':
            support_worker(args)
        else:
            evaluate_worker(args)
        return
    prepare(args)
    if args.stage == 'prepare':
        report(root)
        return
    if args.stage in ('smoke', 'train') and not (root / 'evals/sft/summary.json').is_file():
        parser.error('Complete baseline before any RL update')
    try:
        stream_command([sys.executable, '-u', '-m', 'posttrain_math.rl_pilot', *sys.argv[1:], '--worker'],
                       log_path=root / 'steps' / f'{args.stage}.log',
                       status_path=root / 'steps' / f'{args.stage}.json')
    finally:
        report(root)
        print(f'Review bundle: {package_results(root)}', flush=True)


if __name__ == '__main__':
    main()
