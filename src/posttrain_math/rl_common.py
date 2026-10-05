from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from posttrain_math.artifacts import resolve_local_base_model
from posttrain_math.environment import native_bf16_supported
from posttrain_math.prompting import get_prompt_formatter


def git_commit() -> str | None:
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


def resolve_rl_precision(precision: str) -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for RL training/probing.")
    if precision == "auto":
        return "bf16" if native_bf16_supported() else "fp16"
    if precision not in {"bf16", "fp16", "fp32"}:
        raise ValueError(f"Unknown precision: {precision}")
    if precision == "bf16" and not native_bf16_supported():
        raise RuntimeError("BF16 requested but GPU has no native BF16 support.")
    return precision


def dtype_for_precision(precision: str) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[precision]


def load_sft_adapter(
    model_path: Path,
    *,
    precision: str,
    trainable: bool,
):
    """Load a local LoRA adapter and its pinned base without network access."""
    adapter_config_path = model_path / "adapter_config.json"
    if not adapter_config_path.is_file():
        raise ValueError(
            "RL expects a LoRA SFT adapter/checkpoint as --model. "
            f"Missing: {adapter_config_path}"
        )

    peft_config = PeftConfig.from_pretrained(model_path, local_files_only=True)
    base_model_path = resolve_local_base_model(model_path, str(peft_config.base_model_name_or_path))

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        local_files_only=True,
        dtype=dtype_for_precision(precision),
    )
    base_model.config.use_cache = not trainable
    model = PeftModel.from_pretrained(
        base_model,
        model_path,
        local_files_only=True,
        is_trainable=trainable,
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
    model.generation_config = GenerationConfig(
        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
    )
    base_model.generation_config = model.generation_config
    return model, tokenizer, base_model_path


def build_boxed_numeric_rl_dataset(
    *,
    data_dir: Path,
    tokenizer,
    prompt_name: str,
    max_prompt_length: int,
    limit_prompts: int | None,
    split: str = "train",
) -> tuple[Dataset, dict[str, int]]:
    """Build the shared boxed-numeric-v1 dataset for RL and diagnostics."""
    if max_prompt_length <= 0:
        raise ValueError("max_prompt_length must be positive.")
    if split not in {"train", "dev", "test"}:
        raise ValueError(f"Unknown split: {split}")

    path = data_dir / f"{split}.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"Processed {split} split not found: {path}")

    df = pd.read_parquet(path)
    required = {
        "problem",
        "type",
        "level",
        "numeric_eligible",
        "gt_numerator",
        "gt_denominator",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{split}: missing columns {sorted(missing)}. "
            "Re-run `posttrain-math data prepare`."
        )

    cohort_df = df[df["numeric_eligible"].astype(bool)].copy()
    formatter = get_prompt_formatter(prompt_name)
    records: list[dict[str, str | int]] = []
    overlong_prompt = 0

    for source_index, row in cohort_df.iterrows():
        numerator = int(row["gt_numerator"])
        denominator = int(row["gt_denominator"])
        if denominator == 0:
            raise RuntimeError("Prepared numeric cohort contains a zero denominator.")

        prompt = formatter(str(row["problem"]))
        prompt_length = len(
            tokenizer(prompt, add_special_tokens=False)["input_ids"]
        )
        if prompt_length > max_prompt_length:
            overlong_prompt += 1
            continue

        records.append(
            {
                "prompt_id": f"{split}:{source_index}",
                "prompt": prompt,
                "problem": str(row["problem"]),
                "type": str(row["type"]),
                "level": str(row["level"]),
                "gt_numerator": numerator,
                "gt_denominator": denominator,
            }
        )

    if limit_prompts is not None:
        if limit_prompts <= 0:
            raise ValueError("limit_prompts must be positive.")
        records = records[:limit_prompts]

    if not records:
        raise ValueError(
            f"No boxed-numeric-v1 {split} prompts remain after filtering."
        )

    stats = {
        "source_rows": len(df),
        "cohort_rows": len(cohort_df),
        "ineligible_rows": len(df) - len(cohort_df),
        "overlong_prompt_excluded": overlong_prompt,
        "selected_rows": len(records),
    }
    return Dataset.from_list(records), stats
