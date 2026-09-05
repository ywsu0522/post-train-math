import argparse
from pathlib import Path

from posttrain_math.data import (
    HENDRYCKS_MATH_REPO,
    download_raw_datasets,
    inspect_raw_datasets,
    prepare_datasets,
)
from posttrain_math.environment import (
    inspect_environment,
    print_environment_report,
)
from posttrain_math.evaluation import (
    evaluate,
)
from posttrain_math.modeling import (
    HFModelRunner,
)
from posttrain_math.prompting import (
    PROMPT_STRATEGIES,
    get_prompt_formatter,
)
from posttrain_math.resources import (
    DEFAULT_MODEL_DIR,
    DEFAULT_MODEL_REPO,
    download_model,
)
from posttrain_math.training import (
    inspect_sft_data,
    overfit_one_batch,
    train_sft,
)

DEFAULT_TRAIN_DATASET = Path(
    "data/raw/math_train.parquet"
)

DEFAULT_TEST_DATASET = Path(
    "data/raw/math_test.parquet"
)

DEFAULT_PROCESSED_DIR = Path(
    "data/processed"
)

DEFAULT_MODEL = DEFAULT_MODEL_DIR
DEFAULT_RAW_DATA_DIR = Path("data/raw")


def _add_training_data_args(
    parser: argparse.ArgumentParser,
) -> None:
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
    )

    parser.add_argument(
        "--prompt",
        choices=PROMPT_STRATEGIES,
        default="boxed",
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="posttrain-math",
        description=(
            "Research engineering lab for "
            "mathematical LLM post-training."
        ),
    )

    commands = parser.add_subparsers(
        dest="command",
        required=True,
    )

    # model
    model_parser = commands.add_parser(
        "model",
    )
    model_commands = model_parser.add_subparsers(
        dest="model_command",
        required=True,
    )
    model_download = model_commands.add_parser(
        "download",
    )
    model_download.add_argument(
        "--repo-id",
        default=DEFAULT_MODEL_REPO,
    )
    model_download.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_MODEL,
    )
    model_download.add_argument(
        "--revision",
        default="main",
    )
    model_download.add_argument(
        "--force",
        action="store_true",
    )

    # environment
    env_parser = commands.add_parser(
        "environment",
    )

    env_parser.add_argument(
        "--train-data",
        type=Path,
        default=DEFAULT_TRAIN_DATASET,
    )

    env_parser.add_argument(
        "--test-data",
        type=Path,
        default=DEFAULT_TEST_DATASET,
    )

    env_parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
    )

    # data
    data_parser = commands.add_parser(
        "data",
    )

    data_commands = (
        data_parser.add_subparsers(
            dest="data_command",
            required=True,
        )
    )

    download_parser = data_commands.add_parser(
        "download",
    )
    download_parser.add_argument(
        "--repo-id",
        default=HENDRYCKS_MATH_REPO,
    )
    download_parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RAW_DATA_DIR,
    )
    download_parser.add_argument(
        "--revision",
        default="main",
    )
    download_parser.add_argument(
        "--force",
        action="store_true",
    )

    inspect_parser = (
        data_commands.add_parser(
            "inspect",
        )
    )

    inspect_parser.add_argument(
        "--train-data",
        type=Path,
        default=DEFAULT_TRAIN_DATASET,
    )

    inspect_parser.add_argument(
        "--test-data",
        type=Path,
        default=DEFAULT_TEST_DATASET,
    )

    prepare_parser = (
        data_commands.add_parser(
            "prepare",
        )
    )

    prepare_parser.add_argument(
        "--train-data",
        type=Path,
        default=DEFAULT_TRAIN_DATASET,
    )

    prepare_parser.add_argument(
        "--test-data",
        type=Path,
        default=DEFAULT_TEST_DATASET,
    )

    prepare_parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
    )

    prepare_parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    prepare_parser.add_argument(
        "--dev-ratio",
        type=float,
        default=0.1,
    )

    # eval
    eval_parser = commands.add_parser(
        "eval",
    )

    eval_parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
    )

    eval_parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
    )

    eval_parser.add_argument(
        "--split",
        choices=("dev", "test"),
        default="dev",
    )

    eval_parser.add_argument(
        "--prompt",
        choices=PROMPT_STRATEGIES,
        default="boxed",
    )

    eval_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )

    eval_parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )

    eval_parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
    )

    eval_parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    # train
    train_parser = commands.add_parser(
        "train",
    )

    train_commands = (
        train_parser.add_subparsers(
            dest="train_command",
            required=True,
        )
    )

    # train inspect
    train_inspect = (
        train_commands.add_parser(
            "inspect",
        )
    )

    _add_training_data_args(
        train_inspect
    )

    train_inspect.add_argument(
        "--samples",
        type=int,
        default=3,
    )

    train_inspect.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    # train overfit-one-batch
    overfit_parser = (
        train_commands.add_parser(
            "overfit-one-batch",
        )
    )

    _add_training_data_args(
        overfit_parser
    )

    overfit_parser.add_argument(
        "--overlong-policy",
        choices=(
            "error",
            "drop",
        ),
        default="error",
    )

    overfit_parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
    )

    overfit_parser.add_argument(
        "--steps",
        type=int,
        default=50,
    )

    overfit_parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help=(
            "Default: 2e-4 for LoRA, "
            "1e-4 for full-parameter sanity."
        ),
    )

    overfit_parser.add_argument(
        "--peft",
        choices=("none", "lora"),
        default="lora",
    )

    overfit_parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
    )

    overfit_parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
    )

    overfit_parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
    )

    overfit_parser.add_argument(
        "--lora-target-modules",
        default="all-linear",
    )

    overfit_parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
    )

    overfit_parser.add_argument(
        "--precision",
        choices=(
            "auto",
            "bf16",
            "fp16",
            "fp32",
        ),
        default="auto",
    )

    overfit_parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    overfit_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )

    # train sft
    sft_parser = (
        train_commands.add_parser(
            "sft",
        )
    )

    _add_training_data_args(
        sft_parser
    )

    sft_parser.add_argument(
        "--overlong-policy",
        choices=(
            "error",
            "drop",
        ),
        default="error",
    )

    sft_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )

    sft_parser.add_argument(
        "--epochs",
        type=float,
        default=1.0,
    )

    sft_parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help=(
            "Default: 2e-4 for LoRA, "
            "2e-5 for full-parameter SFT."
        ),
    )

    sft_parser.add_argument(
        "--peft",
        choices=("none", "lora"),
        default="lora",
    )

    sft_parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
    )

    sft_parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
    )

    sft_parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
    )

    sft_parser.add_argument(
        "--lora-target-modules",
        default="all-linear",
        help=(
            "Use 'all-linear' or a comma-separated "
            "module-name list."
        ),
    )

    sft_parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
    )

    sft_parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=2,
    )

    sft_parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=None,
        help=(
            "Explicit per-process accumulation. If omitted, "
            "--global-batch-size is preserved across 1/2 GPUs."
        ),
    )

    sft_parser.add_argument(
        "--global-batch-size",
        type=int,
        default=16,
        help=(
            "Target global effective batch size. Used when "
            "--gradient-accumulation is omitted."
        ),
    )

    sft_parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
    )

    sft_parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.03,
    )

    sft_parser.add_argument(
        "--scheduler",
        choices=(
            "linear",
            "cosine",
            "constant",
        ),
        default="linear",
    )

    sft_parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
    )

    sft_parser.add_argument(
        "--optim",
        choices=(
            "adamw_torch",
            "adamw_torch_fused",
        ),
        default="adamw_torch",
    )

    sft_parser.add_argument(
        "--precision",
        choices=(
            "auto",
            "bf16",
            "fp16",
            "fp32",
        ),
        default="auto",
    )

    sft_parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
    )

    sft_parser.add_argument(
        "--logging-steps",
        type=int,
        default=10,
    )

    sft_parser.add_argument(
        "--eval-steps",
        type=int,
        default=50,
    )

    sft_parser.add_argument(
        "--save-steps",
        type=int,
        default=100,
    )

    sft_parser.add_argument(
        "--save-total-limit",
        type=int,
        default=10,
    )

    sft_parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    sft_parser.add_argument(
        "--resume-from-checkpoint",
        type=Path,
        default=None,
    )

    return parser


def run_model(
    args: argparse.Namespace,
) -> int:
    if args.model_command == "download":
        download_model(
            repo_id=args.repo_id,
            output_dir=args.output_dir,
            revision=args.revision,
            force=args.force,
        )
        return 0

    raise RuntimeError(
        "Unknown model command: "
        f"{args.model_command}"
    )


def run_environment(
    args: argparse.Namespace,
) -> int:
    report = inspect_environment(
        train_dataset_path=(
            args.train_data
        ),
        test_dataset_path=(
            args.test_data
        ),
        model_path=args.model,
    )

    print_environment_report(
        report,
        train_dataset_path=(
            args.train_data
        ),
        test_dataset_path=(
            args.test_data
        ),
        model_path=args.model,
    )

    return 0 if report.ok else 1


def run_data(
    args: argparse.Namespace,
) -> int:
    if args.data_command == "download":
        download_raw_datasets(
            output_dir=args.output_dir,
            repo_id=args.repo_id,
            revision=args.revision,
            force=args.force,
        )
        return 0

    if args.data_command == "inspect":
        inspect_raw_datasets(
            train_path=args.train_data,
            test_path=args.test_data,
        )
        return 0

    if args.data_command == "prepare":
        prepare_datasets(
            train_path=args.train_data,
            test_path=args.test_data,
            output_dir=args.output_dir,
            seed=args.seed,
            dev_ratio=args.dev_ratio,
        )
        return 0

    raise RuntimeError(
        "Unknown data command: "
        f"{args.data_command}"
    )


def run_eval(
    args: argparse.Namespace,
) -> int:
    prompt_formatter = (
        get_prompt_formatter(
            args.prompt
        )
    )

    output_dir = args.output_dir

    if output_dir is None:
        output_dir = Path(
            "runs/"
            f"eval-{args.model.name}-"
            f"{args.split}-"
            f"{args.prompt}"
        )

    generator = (
        HFModelRunner.from_pretrained(
            args.model,
            batch_size=args.batch_size,
        )
    )

    evaluate(
        generator,
        prompt_formatter=(
            prompt_formatter
        ),
        prompt_name=args.prompt,
        data_dir=args.data_dir,
        split=args.split,
        output_dir=output_dir,
        max_new_tokens=(
            args.max_new_tokens
        ),
        limit=args.limit,
    )

    return 0


def run_train(
    args: argparse.Namespace,
) -> int:
    if args.train_command == "inspect":
        inspect_sft_data(
            model_path=args.model,
            data_dir=args.data_dir,
            prompt_name=args.prompt,
            max_length=args.max_length,
            samples=args.samples,
            seed=args.seed,
        )

        return 0

    if (
        args.train_command
        == "overfit-one-batch"
    ):
        output_dir = (
            args.output_dir
        )

        if output_dir is None:
            output_dir = Path(
                "runs/"
                f"sanity-{args.peft}-{args.model.name}-"
                f"{args.prompt}"
            )

        learning_rate = args.learning_rate
        if learning_rate is None:
            learning_rate = (
                2e-4
                if args.peft == "lora"
                else 1e-4
            )

        overfit_one_batch(
            model_path=args.model,
            data_dir=args.data_dir,
            prompt_name=args.prompt,
            max_length=args.max_length,
            overlong_policy=(
                args.overlong_policy
            ),
            batch_size=args.batch_size,
            steps=args.steps,
            learning_rate=learning_rate,
            max_grad_norm=(
                args.max_grad_norm
            ),
            precision=args.precision,
            peft_method=args.peft,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            lora_target_modules=(
                args.lora_target_modules
            ),
            seed=args.seed,
            output_dir=output_dir,
        )

        return 0

    if args.train_command == "sft":
        output_dir = (
            args.output_dir
        )

        if output_dir is None:
            output_dir = Path(
                "runs/"
                f"sft-{args.peft}-{args.model.name}-"
                f"{args.prompt}"
            )

        learning_rate = args.learning_rate
        if learning_rate is None:
            learning_rate = (
                2e-4
                if args.peft == "lora"
                else 2e-5
            )

        train_sft(
            model_path=args.model,
            data_dir=args.data_dir,
            prompt_name=args.prompt,
            max_length=args.max_length,
            overlong_policy=(
                args.overlong_policy
            ),
            output_dir=output_dir,
            epochs=args.epochs,
            learning_rate=learning_rate,
            batch_size=args.batch_size,
            eval_batch_size=(
                args.eval_batch_size
            ),
            gradient_accumulation=(
                args.gradient_accumulation
            ),
            global_batch_size=(
                args.global_batch_size
            ),
            weight_decay=(
                args.weight_decay
            ),
            warmup_ratio=(
                args.warmup_ratio
            ),
            scheduler=args.scheduler,
            max_grad_norm=(
                args.max_grad_norm
            ),
            optim=args.optim,
            precision=args.precision,
            peft_method=args.peft,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            lora_target_modules=(
                args.lora_target_modules
            ),
            gradient_checkpointing=(
                args.gradient_checkpointing
            ),
            logging_steps=(
                args.logging_steps
            ),
            eval_steps=args.eval_steps,
            save_steps=args.save_steps,
            save_total_limit=(
                args.save_total_limit
            ),
            seed=args.seed,
            resume_from_checkpoint=(
                args.resume_from_checkpoint
            ),
        )

        return 0

    raise RuntimeError(
        "Unknown train command: "
        f"{args.train_command}"
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.command == "model":
            exit_code = run_model(args)

        elif args.command == "environment":
            exit_code = (
                run_environment(args)
            )

        elif args.command == "data":
            exit_code = run_data(args)

        elif args.command == "eval":
            exit_code = run_eval(args)

        elif args.command == "train":
            exit_code = run_train(args)

        else:
            parser.error(
                "Unknown command"
            )

    except (
        FileNotFoundError,
        FileExistsError,
        ValueError,
        RuntimeError,
        FloatingPointError,
        AssertionError,
    ) as exc:
        parser.exit(
            status=1,
            message=f"ERROR: {exc}\n",
        )

    raise SystemExit(exit_code)
