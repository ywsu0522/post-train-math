"""The project exposes only frozen data preparation and candidate evaluation."""

import argparse
import hashlib
import json
from pathlib import Path

from post_train_math.baselines import MODELS, make_chat_prompt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="post-train-math")
    commands = parser.add_subparsers(dest="command", required=True)
    data = commands.add_parser("data").add_subparsers(dest="data_command", required=True)
    download = data.add_parser("download", help="Download the pinned MATH source.")
    download.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    download.add_argument("--force", action="store_true")
    prepare = data.add_parser("prepare", help="Verify source and prepare the fixed cohort.")
    prepare.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    prepare.add_argument("--output-dir", type=Path, default=Path("data/processed"))

    model = commands.add_parser("model").add_subparsers(dest="model_command", required=True)
    download_model = model.add_parser("download")
    download_model.add_argument("--model-key", choices=tuple(MODELS), required=True)
    download_model.add_argument("--output-dir", type=Path)

    evaluate = commands.add_parser("eval", help="Evaluate a pinned candidate on boxed-numeric-v2 dev.")
    evaluate.add_argument("--model-key", choices=tuple(MODELS), required=True)
    evaluate.add_argument("--model-dir", type=Path)
    evaluate.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--batch-size", type=int, default=1)
    evaluate.add_argument("--max-new-tokens", type=int, default=1024)
    evaluate.add_argument("--limit", type=int, help="Smoke evaluation only; omit for the full cohort.")
    return parser


def run_eval(args: argparse.Namespace) -> None:
    from post_train_math.baselines import (
        validate_model_source,
        validate_processed_manifest,
    )
    from post_train_math.data import COHORT_NAME
    from post_train_math.evaluation import evaluate
    from post_train_math.modeling import HFModelRunner

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be positive")
    if args.batch_size <= 0 or args.max_new_tokens <= 0:
        raise ValueError("batch-size and max-new-tokens must be positive")
    spec = MODELS[args.model_key]
    model_dir = args.model_dir or Path(spec.local_dir)
    manifest, manifest_hash = validate_processed_manifest(args.data_dir)
    source = validate_model_source(model_dir, spec)
    runner = HFModelRunner.from_pretrained(model_dir, batch_size=args.batch_size)
    metrics = evaluate(
        runner,
        prompt_formatter=lambda problem: make_chat_prompt(runner.tokenizer, spec, problem),
        prompt_name=f"chat-{spec.key}-boxed",
        data_dir=args.data_dir,
        split="dev",
        output_dir=args.output_dir,
        max_new_tokens=args.max_new_tokens,
        limit=args.limit,
    )
    if runner.rank != 0:
        return
    report = {
        "model_key": spec.key,
        "model_repo": spec.repo_id,
        "model_revision": spec.revision,
        "model_source": source,
        "dataset_split": "dev",
        "cohort": COHORT_NAME,
        "data_manifest_sha256": manifest_hash,
        "dev_cohort": manifest["cohort"]["dev"],
        "system_prompt": spec.system_prompt,
        "chat_template_sha256": hashlib.sha256(runner.tokenizer.chat_template.encode()).hexdigest(),
        "generation": {
            "do_sample": False,
            "max_new_tokens": args.max_new_tokens,
            "batch_size": args.batch_size,
            "limit": args.limit,
        },
        "verifier_metrics": metrics,
    }
    (args.output_dir / "baseline_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.command == "data":
            from post_train_math.data import download_raw_datasets, prepare_datasets

            if args.data_command == "download":
                download_raw_datasets(output_dir=args.output_dir, force=args.force)
            else:
                prepare_datasets(
                    args.raw_dir / "math_train.parquet",
                    args.raw_dir / "math_test.parquet",
                    args.output_dir,
                    seed=42,
                    dev_ratio=0.1,
                )
        elif args.command == "model":
            from post_train_math.resources import download_model

            spec = MODELS[args.model_key]
            download_model(
                repo_id=spec.repo_id,
                revision=spec.revision,
                output_dir=args.output_dir or Path(spec.local_dir),
            )
        else:
            run_eval(args)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")
