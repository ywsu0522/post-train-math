from __future__ import annotations

import argparse
import json
import subprocess
from fractions import Fraction
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from datasets import Dataset
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from posttrain_math.answers import extract_final_boxed_rational
from posttrain_math.distributed import (
    get_distributed_context,
    resolve_gradient_accumulation,
)
from posttrain_math.environment import native_bf16_supported
from posttrain_math.prompting import PROMPT_STRATEGIES, get_prompt_formatter


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _resolve_precision(precision: str) -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for GRPO.")
    if precision == "auto":
        return "bf16" if native_bf16_supported() else "fp16"
    if precision not in {"bf16", "fp16", "fp32"}:
        raise ValueError(f"Unknown precision: {precision}")
    if precision == "bf16" and not native_bf16_supported():
        raise RuntimeError("BF16 requested but GPU has no native BF16 support.")
    return precision


def _dtype_for_precision(precision: str) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[precision]


def _completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts: list[str] = []
        for item in completion:
            if isinstance(item, dict) and "content" in item:
                parts.append(str(item["content"]))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    if isinstance(completion, dict) and "content" in completion:
        return str(completion["content"])
    return str(completion)


def make_exact_rational_reward():
    """Binary RLVR reward shared by all exact-rational-v1 RL methods."""

    def exact_rational_reward(
        completions,
        gt_numerator,
        gt_denominator,
        **kwargs,
    ) -> list[float]:
        del kwargs
        rewards: list[float] = []
        for completion, numerator, denominator in zip(
            completions,
            gt_numerator,
            gt_denominator,
            strict=True,
        ):
            denominator = int(denominator)
            if denominator == 0:
                raise RuntimeError("RL dataset contains a zero gold denominator.")
            gold = Fraction(int(numerator), denominator)
            prediction = extract_final_boxed_rational(_completion_text(completion))
            rewards.append(float(prediction == gold))
        return rewards

    exact_rational_reward.__name__ = "exact_rational_reward"
    return exact_rational_reward


def _load_trainable_sft_adapter(
    model_path: Path,
    *,
    precision: str,
):
    adapter_config_path = model_path / "adapter_config.json"
    if not adapter_config_path.is_file():
        raise ValueError(
            "GRPO expects a LoRA SFT adapter/checkpoint as --model. "
            f"Missing: {adapter_config_path}"
        )

    peft_config = PeftConfig.from_pretrained(model_path, local_files_only=True)
    base_model_path = Path(peft_config.base_model_name_or_path)
    if not base_model_path.is_dir():
        raise FileNotFoundError(
            "The SFT adapter points to a base model directory that is "
            f"not available locally: {base_model_path}"
        )

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        local_files_only=True,
        dtype=_dtype_for_precision(precision),
    )
    base_model.config.use_cache = False
    model = PeftModel.from_pretrained(
        base_model,
        model_path,
        local_files_only=True,
        is_trainable=True,
    )

    tokenizer_source = (
        model_path
        if (model_path / "tokenizer_config.json").is_file()
        else base_model_path
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        local_files_only=True,
    )
    if tokenizer.eos_token_id is None:
        raise RuntimeError("Tokenizer has no EOS token.")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return model, tokenizer, base_model_path


def _build_grpo_dataset(
    *,
    data_dir: Path,
    tokenizer,
    prompt_name: str,
    max_prompt_length: int,
    limit_prompts: int | None,
) -> tuple[Dataset, dict[str, int]]:
    if max_prompt_length <= 0:
        raise ValueError("max_prompt_length must be positive.")

    path = data_dir / "train.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"Processed training split not found: {path}")

    df = pd.read_parquet(path)
    required = {
        "problem",
        "rational_eligible",
        "gt_numerator",
        "gt_denominator",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"train: missing columns {sorted(missing)}. "
            "Re-run `posttrain-math data prepare`."
        )

    cohort_df = df[df["rational_eligible"].astype(bool)].copy()
    formatter = get_prompt_formatter(prompt_name)
    records: list[dict[str, str | int]] = []
    overlong_prompt = 0

    for row in cohort_df.itertuples(index=False):
        numerator = int(row.gt_numerator)
        denominator = int(row.gt_denominator)
        if denominator == 0:
            raise RuntimeError("Prepared rational cohort contains a zero denominator.")

        prompt = formatter(str(row.problem))
        prompt_length = len(
            tokenizer(prompt, add_special_tokens=False)["input_ids"]
        )
        if prompt_length > max_prompt_length:
            overlong_prompt += 1
            continue

        records.append(
            {
                "prompt": prompt,
                "gt_numerator": numerator,
                "gt_denominator": denominator,
            }
        )

    if limit_prompts is not None:
        if limit_prompts <= 0:
            raise ValueError("limit_prompts must be positive.")
        records = records[:limit_prompts]

    if not records:
        raise ValueError("No exact-rational-v1 GRPO prompts remain after filtering.")

    stats = {
        "source_rows": len(df),
        "cohort_rows": len(cohort_df),
        "ineligible_rows": len(df) - len(cohort_df),
        "overlong_prompt_excluded": overlong_prompt,
        "training_rows": len(records),
    }
    return Dataset.from_list(records), stats


def train_grpo(
    *,
    model_path: Path,
    data_dir: Path,
    prompt_name: str,
    output_dir: Path,
    max_steps: int,
    learning_rate: float,
    per_device_batch_size: int,
    gradient_accumulation: int | None,
    global_batch_size: int,
    num_generations: int,
    max_prompt_length: int,
    max_completion_length: int,
    temperature: float,
    top_p: float,
    beta: float,
    precision: str,
    gradient_checkpointing: bool,
    logging_steps: int,
    save_steps: int,
    save_total_limit: int,
    seed: int,
    limit_prompts: int | None,
    resume_from_checkpoint: Path | None,
) -> None:
    if max_steps <= 0:
        raise ValueError("max_steps must be positive.")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")
    if per_device_batch_size <= 0:
        raise ValueError("per_device_batch_size must be positive.")
    if num_generations <= 1:
        raise ValueError("num_generations must be greater than 1.")
    if max_completion_length <= 0:
        raise ValueError("max_completion_length must be positive.")
    if not 0.0 < temperature:
        raise ValueError("temperature must be positive.")
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in (0, 1].")
    if beta < 0.0:
        raise ValueError("beta must be non-negative.")
    if min(logging_steps, save_steps, save_total_limit) <= 0:
        raise ValueError(
            "logging_steps, save_steps and save_total_limit must be positive."
        )

    context = get_distributed_context()
    precision = _resolve_precision(precision)
    resolved_accumulation, effective_batch = resolve_gradient_accumulation(
        per_device_batch_size=per_device_batch_size,
        world_size=context.world_size,
        gradient_accumulation=gradient_accumulation,
        global_batch_size=global_batch_size,
    )
    if effective_batch % num_generations != 0:
        raise ValueError(
            "GRPO effective global batch must be divisible by "
            f"num_generations: {effective_batch} % {num_generations} != 0"
        )
    prompt_groups_per_update = effective_batch // num_generations

    model, tokenizer, base_model_path = _load_trainable_sft_adapter(
        model_path,
        precision=precision,
    )
    train_dataset, data_stats = _build_grpo_dataset(
        data_dir=data_dir,
        tokenizer=tokenizer,
        prompt_name=prompt_name,
        max_prompt_length=max_prompt_length,
        limit_prompts=limit_prompts,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        "algorithm": "GRPO",
        "variant": "original-grpo-loss",
        "git_commit": _git_commit(),
        "model": str(model_path),
        "base_model": str(base_model_path),
        "data_dir": str(data_dir),
        "cohort": "exact-rational-v1",
        "prompt": prompt_name,
        "data": data_stats,
        "max_steps": max_steps,
        "learning_rate": learning_rate,
        "per_device_batch_size": per_device_batch_size,
        "world_size": context.world_size,
        "gradient_accumulation": resolved_accumulation,
        "global_effective_batch_size": effective_batch,
        "num_generations": num_generations,
        "prompt_groups_per_update": prompt_groups_per_update,
        "max_prompt_length": max_prompt_length,
        "max_completion_length": max_completion_length,
        "temperature": temperature,
        "top_p": top_p,
        "reward": {
            "verifier": "exact-rational-v1",
            "correct": 1.0,
            "otherwise": 0.0,
        },
        "grpo": {
            "loss_type": "grpo",
            "num_iterations": 1,
            "scale_rewards": "group",
        },
        "beta": beta,
        "precision": precision,
        "gradient_checkpointing": gradient_checkpointing,
        "logging_steps": logging_steps,
        "save_steps": save_steps,
        "save_total_limit": save_total_limit,
        "seed": seed,
        "torch": torch.__version__,
        "transformers": version("transformers"),
        "trl": version("trl"),
        "peft": version("peft"),
        "datasets": version("datasets"),
    }

    if context.is_main_process:
        (output_dir / "run_config.json").write_text(
            json.dumps(run_config, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print("GRPO training plan")
        print(f"- SFT adapter: {model_path}")
        print(f"- base model: {base_model_path}")
        print(f"- cohort prompts: {len(train_dataset)}")
        print(f"- world size: {context.world_size}")
        print(f"- per-device batch: {per_device_batch_size}")
        print(f"- gradient accumulation: {resolved_accumulation}")
        print(f"- global effective batch: {effective_batch}")
        print(f"- generations per prompt: {num_generations}")
        print(f"- prompt groups per update: {prompt_groups_per_update}")
        print(f"- max completion length: {max_completion_length}")
        print(f"- max steps: {max_steps}")
        print(f"- learning rate: {learning_rate}")
        print("- reward: exact-rational-v1 binary {0,1}")
        print("- GRPO loss variant: original sequence-normalized GRPO")
        print(f"- KL beta: {beta}")
        print(f"- precision: {precision}")
        print()

    args = GRPOConfig(
        output_dir=str(output_dir),
        max_steps=max_steps,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=resolved_accumulation,
        max_grad_norm=1.0,
        optim="adamw_torch",
        bf16=(precision == "bf16"),
        fp16=(precision == "fp16"),
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs=(
            {"use_reentrant": False} if gradient_checkpointing else None
        ),
        logging_strategy="steps",
        logging_steps=logging_steps,
        logging_first_step=True,
        logging_nan_inf_filter=False,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        report_to="none",
        seed=seed,
        data_seed=seed,
        disable_tqdm=True,
        remove_unused_columns=False,
        log_on_each_node=False,
        ddp_find_unused_parameters=(False if context.world_size > 1 else None),
        num_generations=num_generations,
        num_iterations=1,
        scale_rewards="group",
        loss_type="grpo",
        max_completion_length=max_completion_length,
        temperature=temperature,
        top_p=top_p,
        top_k=0,
        repetition_penalty=1.0,
        beta=beta,
        disable_dropout=True,
        use_vllm=False,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=make_exact_rational_reward(),
        args=args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    train_result = trainer.train(
        resume_from_checkpoint=(
            str(resume_from_checkpoint)
            if resume_from_checkpoint is not None
            else None
        )
    )

    final_model_dir = output_dir / "final-model"
    trainer.save_model(str(final_model_dir))
    trainer.accelerator.wait_for_everyone()

    if context.is_main_process:
        tokenizer.save_pretrained(final_model_dir)
        checkpoints = sorted(
            (
                path.name
                for path in output_dir.glob("checkpoint-*")
                if path.is_dir()
            ),
            key=lambda name: int(name.rsplit("-", 1)[1]),
        )
        summary = {
            "train_metrics": train_result.metrics,
            "checkpoints": checkpoints,
            "final_model": str(final_model_dir),
            "log_history": trainer.state.log_history,
        }
        (output_dir / "train_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )
        print()
        print("GRPO complete")
        print(f"- checkpoints: {len(checkpoints)}")
        print(f"- final model: {final_model_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="posttrain-math-grpo",
        description=(
            "GRPO post-training for a LoRA SFT checkpoint using the original "
            "sequence-normalized GRPO loss and exact-rational-v1 binary rewards."
        ),
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="LoRA SFT checkpoint or final-model directory.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--prompt", choices=PROMPT_STRATEGIES, default="boxed")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/olmo2-1b-grpo-v1"),
    )
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=None,
        help=(
            "Explicit per-process gradient accumulation. "
            "If omitted, --global-batch-size is preserved across 1/2 GPUs."
        ),
    )
    parser.add_argument("--global-batch-size", type=int, default=4)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--max-completion-length", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument(
        "--beta",
        type=float,
        default=0.0,
        help=(
            "KL coefficient. The baseline defaults to 0 to avoid a reference-model "
            "copy and keep T4 memory use bounded."
        ),
    )
    parser.add_argument(
        "--precision",
        choices=("auto", "bf16", "fp16", "fp32"),
        default="auto",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--save-total-limit", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-prompts", type=int, default=None)
    parser.add_argument("--resume-from-checkpoint", type=Path, default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        train_grpo(
            model_path=args.model,
            data_dir=args.data_dir,
            prompt_name=args.prompt,
            output_dir=args.output_dir,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            per_device_batch_size=args.batch_size,
            gradient_accumulation=args.gradient_accumulation,
            global_batch_size=args.global_batch_size,
            num_generations=args.num_generations,
            max_prompt_length=args.max_prompt_length,
            max_completion_length=args.max_completion_length,
            temperature=args.temperature,
            top_p=args.top_p,
            beta=args.beta,
            precision=args.precision,
            gradient_checkpointing=args.gradient_checkpointing,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            save_total_limit=args.save_total_limit,
            seed=args.seed,
            limit_prompts=args.limit_prompts,
            resume_from_checkpoint=args.resume_from_checkpoint,
        )
    except (FileNotFoundError, ValueError, RuntimeError, FloatingPointError) as exc:
        parser.exit(status=1, message=f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
