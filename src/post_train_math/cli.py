"""Prepare data, train policies, and evaluate saved checkpoints."""

import argparse
import json
from pathlib import Path

from post_train_math.models import MODELS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="post-train-math")
    commands = parser.add_subparsers(dest="command", required=True)
    data = commands.add_parser("data").add_subparsers(
        dest="data_command", required=True
    )
    download = data.add_parser("download", help="Download the pinned raw MATH source")
    download.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    download.add_argument("--force", action="store_true")
    prepare = data.add_parser(
        "prepare", help="Apply boxed-numeric within the original train/test splits"
    )
    prepare.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    prepare.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    model = commands.add_parser("model").add_subparsers(
        dest="model_command", required=True
    )
    model_download = model.add_parser("download")
    model_download.add_argument("--model-key", choices=tuple(MODELS), required=True)
    model_download.add_argument("--output-dir", type=Path)
    for name in ("eval", "eval-series", "train"):
        command = commands.add_parser(name)
        command.add_argument("--model-key", choices=tuple(MODELS), required=True)
        command.add_argument("--model-dir", type=Path)
        command.add_argument("--data-dir", type=Path, default=Path("data/processed"))
        command.add_argument("--output-dir", type=Path, required=True)
        if name in ("eval", "eval-series"):
            command.add_argument("--batch-size", type=int, default=1)
            command.add_argument(
                "--test-set", choices=("fixed512", "full"), default="fixed512"
            )
            command.add_argument(
                "--series-name", help="Label for method comparison plots"
            )
            if name == "eval":
                command.add_argument("--adapter-dir", type=Path)
                command.add_argument("--limit", type=int, help="Smoke evaluation only")
            else:
                command.add_argument("--run-dir", type=Path, required=True)
                command.add_argument(
                    "--steps", type=int, nargs="+", default=[50, 100, 150, 200]
                )
                command.add_argument(
                    "--include-base",
                    action=argparse.BooleanOptionalAction,
                    default=True,
                )
        else:
            command.add_argument(
                "--prompts-per-update", type=int, default=8, help="Global B"
            )
            command.add_argument(
                "--rollouts", type=int, default=4, help="R per prompt, >= 2"
            )
            command.add_argument(
                "--per-device-train-batch-size",
                type=int,
                default=1,
                help="Sequences/device",
            )
            command.add_argument(
                "--gradient-accumulation-steps",
                type=int,
                default=32,
                help="Optimizer accumulation steps; B*R must equal per-device-train-batch-size*world-size*steps",
            )
            command.add_argument(
                "--prepare-only",
                action="store_true",
                help="Save the model-specific training subset on CPU without training",
            )
            command.add_argument("--max-steps", type=int, default=200)
            command.add_argument("--learning-rate", type=float, default=1e-5)
            command.add_argument(
                "--resume-from-checkpoint",
                type=Path,
                help="Resume a complete checkpoint-N, including optimizer and RNG state",
            )
    plot = commands.add_parser("plot", help="Plot saved evaluation reports on CPU")
    plot.add_argument("--reports", type=Path, nargs="+", required=True)
    plot.add_argument("--output-dir", type=Path, required=True)
    return parser


def run_eval(args) -> dict:
    import torch
    from transformers import AutoModelForCausalLM

    from post_train_math.data import load_cohort, validate_processed_manifest
    from post_train_math.evaluation import (
        METRIC_DEFINITIONS,
        adapter_metadata,
        score_completion,
    )
    from post_train_math.models import (
        load_tokenizer,
        make_chat_prompt,
        validate_model_source,
    )
    from post_train_math.rloo import MAX_COMPLETION_LENGTH, MAX_PROMPT_LENGTH
    from post_train_math.testing import select_testing_set

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        raise ValueError("batch-size and limit must be positive")
    manifest, manifest_hash = validate_processed_manifest(args.data_dir)
    spec = MODELS[args.model_key]
    model_dir = args.model_dir or Path(spec.local_dir)
    source = validate_model_source(model_dir, spec)
    checkpoint = adapter_metadata(args.adapter_dir, spec, manifest_hash)
    frame, test_set = select_testing_set(
        load_cohort(args.data_dir, "test"), args.test_set
    )
    if args.limit is not None:
        frame = frame.head(args.limit)
    if frame.empty:
        raise ValueError("Testing set is empty")
    if not torch.cuda.is_available():
        raise RuntimeError("Model evaluation requires CUDA")
    tokenizer = load_tokenizer(model_dir)
    tokenizer.truncation_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    ).to("cuda")
    if args.adapter_dir:
        from peft import PeftModel

        model = PeftModel.from_pretrained(
            model, args.adapter_dir, local_files_only=True
        )
    model.eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    correct = boxed = numeric_boxed = 0
    with (args.output_dir / "predictions.jsonl").open("w", encoding="utf-8") as output:
        for start in range(0, len(frame), args.batch_size):
            rows = list(
                frame.iloc[start : start + args.batch_size].itertuples(index=False)
            )
            prompts = [make_chat_prompt(tokenizer, spec, row.problem) for row in rows]
            encoded = tokenizer(
                prompts,
                add_special_tokens=False,
                padding=True,
                truncation=True,
                max_length=MAX_PROMPT_LENGTH,
                return_tensors="pt",
            ).to("cuda")
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=MAX_COMPLETION_LENGTH,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            texts = tokenizer.batch_decode(
                generated[:, encoded["input_ids"].shape[1] :], skip_special_tokens=True
            )
            for row, text in zip(rows, texts, strict=True):
                scores = score_completion(
                    text,
                    numerator=int(row.gt_numerator),
                    denominator=int(row.gt_denominator),
                )
                correct += int(scores["correct"])
                boxed += int(scores["boxed_output"])
                numeric_boxed += int(scores["numeric_boxed_output"])
                output.write(
                    json.dumps(
                        {
                            "problem_id": row.problem_id,
                            "test_cohort_index": int(row.test_cohort_index),
                            "problem": row.problem,
                            "completion": text,
                            "gt_numerator": int(row.gt_numerator),
                            "gt_denominator": int(row.gt_denominator),
                            **scores,
                        }
                    )
                    + "\n"
                )
    report = {
        "schema_version": 2,
        "model_key": spec.key,
        "series_name": args.series_name or spec.key,
        "model_source": source,
        "adapter_dir": str(args.adapter_dir) if args.adapter_dir else None,
        "data_manifest_sha256": manifest_hash,
        "cohort": manifest["cohort"]["name"],
        "split": "test",
        "test_set": test_set,
        "metric_definitions": METRIC_DEFINITIONS,
        **checkpoint,
        "limit": args.limit,
        "total": len(frame),
        "correct": correct,
        "accuracy": correct / len(frame) if len(frame) else 0.0,
        "boxed_outputs": boxed,
        "boxed_output_rate": boxed / len(frame),
        "numeric_boxed_outputs": numeric_boxed,
        "numeric_boxed_output_rate": numeric_boxed / len(frame),
        "do_sample": False,
        "max_prompt_length": MAX_PROMPT_LENGTH,
        "max_completion_length": MAX_COMPLETION_LENGTH,
    }
    (args.output_dir / "eval_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "model_key",
                    "rollout_iterations",
                    "total",
                    "accuracy",
                    "boxed_output_rate",
                    "numeric_boxed_output_rate",
                )
            },
            indent=2,
        )
    )
    return report


def run_eval_series(args) -> None:
    from post_train_math.curves import plot_reports
    from post_train_math.data import validate_processed_manifest
    from post_train_math.evaluation import adapter_metadata

    if args.steps != sorted(set(args.steps)) or any(step < 1 for step in args.steps):
        raise ValueError("Checkpoint steps must be positive, unique, and increasing")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    _, manifest_hash = validate_processed_manifest(args.data_dir)
    for step in args.steps:
        path = args.run_dir / f"checkpoint-{step}"
        metadata = adapter_metadata(path, MODELS[args.model_key], manifest_hash)
        if metadata["rollout_iterations"] != step:
            raise ValueError(
                f"Checkpoint does not match the requested rollout iteration: {path}"
            )
    reports = []
    for step in ([0] if args.include_base else []) + args.steps:
        point = argparse.Namespace(**vars(args))
        point.adapter_dir = args.run_dir / f"checkpoint-{step}" if step else None
        point.output_dir = args.output_dir / f"step-{step:06d}"
        point.series_name = args.series_name or args.run_dir.name
        point.limit = None
        run_eval(point)
        reports.append(point.output_dir / "eval_report.json")
    plot_reports(reports, args.output_dir / "plots")


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
                )
        elif args.command == "model":
            from post_train_math.models import download_model

            spec = MODELS[args.model_key]
            download_model(
                repo_id=spec.repo_id,
                revision=spec.revision,
                output_dir=args.output_dir or Path(spec.local_dir),
            )
        elif args.command == "eval":
            run_eval(args)
        elif args.command == "eval-series":
            run_eval_series(args)
        elif args.command == "plot":
            from post_train_math.curves import plot_reports

            plot_reports(args.reports, args.output_dir)
        else:
            from post_train_math.rloo import run_train

            run_train(args)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
