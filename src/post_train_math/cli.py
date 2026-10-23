"""Four workflows: data, model, eval, train."""

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
        "prepare", help="Split raw data, then apply boxed-numeric"
    )
    prepare.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    prepare.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    model = commands.add_parser("model").add_subparsers(
        dest="model_command", required=True
    )
    model_download = model.add_parser("download")
    model_download.add_argument("--model-key", choices=tuple(MODELS), required=True)
    model_download.add_argument("--output-dir", type=Path)
    for name in ("eval", "train"):
        command = commands.add_parser(name)
        command.add_argument("--model-key", choices=tuple(MODELS), required=True)
        command.add_argument("--model-dir", type=Path)
        command.add_argument("--data-dir", type=Path, default=Path("data/processed"))
        command.add_argument("--output-dir", type=Path, required=True)
        if name == "eval":
            command.add_argument("--split", choices=("dev", "test"), default="dev")
            command.add_argument("--adapter-dir", type=Path)
            command.add_argument("--batch-size", type=int, default=1)
            command.add_argument("--limit", type=int, help="Smoke evaluation only")
        else:
            command.add_argument(
                "--prompts-per-update", type=int, default=8, help="Global B"
            )
            command.add_argument(
                "--rollouts", type=int, default=4, help="R per prompt, >= 2"
            )
            command.add_argument(
                "--micro-batch", type=int, default=1, help="Sequences/device"
            )
            command.add_argument(
                "--gradient-accumulation-steps",
                type=int,
                default=32,
                help="Optimizer accumulation steps; B*R must equal micro-batch*world-size*steps",
            )
            command.add_argument(
                "--prepare-only",
                action="store_true",
                help="Save the model-specific training subset on CPU without training",
            )
            command.add_argument("--max-steps", type=int, default=100)
            command.add_argument("--learning-rate", type=float, default=1e-5)
    return parser


def run_eval(args) -> None:
    import torch
    from transformers import AutoModelForCausalLM

    from post_train_math.answers import verify_boxed_numeric
    from post_train_math.data import load_cohort, validate_processed_manifest
    from post_train_math.models import (
        load_tokenizer,
        make_chat_prompt,
        validate_model_source,
    )
    from post_train_math.rloo import MAX_COMPLETION_LENGTH, MAX_PROMPT_LENGTH

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        raise ValueError("batch-size and limit must be positive")
    manifest, manifest_hash = validate_processed_manifest(args.data_dir)
    spec = MODELS[args.model_key]
    model_dir = args.model_dir or Path(spec.local_dir)
    source = validate_model_source(model_dir, spec)
    if args.adapter_dir:
        report = json.loads((args.adapter_dir.parent / "train_report.json").read_text())
        if (
            report["model_source"] != source
            or report["data_manifest_sha256"] != manifest_hash
        ):
            raise ValueError("Adapter does not match the pinned model and data")
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
    frame = load_cohort(args.data_dir, args.split)
    if args.limit is not None:
        frame = frame.head(args.limit)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    correct = 0
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
                passed = verify_boxed_numeric(
                    text,
                    numerator=int(row.gt_numerator),
                    denominator=int(row.gt_denominator),
                )
                correct += int(passed)
                output.write(
                    json.dumps(
                        {
                            "problem": row.problem,
                            "completion": text,
                            "gt_numerator": int(row.gt_numerator),
                            "gt_denominator": int(row.gt_denominator),
                            "correct": passed,
                        }
                    )
                    + "\n"
                )
    report = {
        "model_key": spec.key,
        "model_source": source,
        "adapter_dir": str(args.adapter_dir) if args.adapter_dir else None,
        "data_manifest_sha256": manifest_hash,
        "cohort": manifest["cohort"]["name"],
        "split": args.split,
        "limit": args.limit,
        "total": len(frame),
        "correct": correct,
        "accuracy": correct / len(frame) if len(frame) else 0.0,
        "do_sample": False,
        "max_prompt_length": MAX_PROMPT_LENGTH,
        "max_completion_length": MAX_COMPLETION_LENGTH,
    }
    (args.output_dir / "eval_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


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
        else:
            from post_train_math.rloo import run_train

            run_train(args)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
