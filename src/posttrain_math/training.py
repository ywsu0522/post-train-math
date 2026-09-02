from __future__ import annotations

import json
import math
import os
import random
import subprocess
import sys
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

# Colab may export its notebook-only inline backend into child processes. The
# training CLI only writes plots to files, so select the headless backend before
# importing Matplotlib and before it reads MPLBACKEND.
os.environ["MPLBACKEND"] = "Agg"
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from tabulate import tabulate
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    TrainerCallback,
)
from trl import SFTConfig, SFTTrainer

from posttrain_math.environment import native_bf16_supported
from posttrain_math.prompting import PromptFormatter, get_prompt_formatter

IGNORE_INDEX = -100
GIB = 1024**3


@dataclass(frozen=True)
class EncodedSFTExample:
    row_id: int
    prompt: str
    completion: str
    input_ids: list[int]
    labels: list[int]
    prompt_length: int
    target_length: int

    @property
    def full_length(self) -> int:
        return len(self.input_ids)

    def as_record(self) -> dict[str, list[int]]:
        return {
            "input_ids": self.input_ids,
            "attention_mask": [1] * self.full_length,
            "labels": self.labels,
        }


# ---------------------------------------------------------------------------
# SFT data contract
# ---------------------------------------------------------------------------


def load_training_tokenizer(model_path: Path):
    if not model_path.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.eos_token_id is None:
        raise RuntimeError("Tokenizer has no EOS token.")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def _load_split(data_dir: Path, split: str) -> pd.DataFrame:
    path = data_dir / f"{split}.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"Processed split not found: {path}")

    df = pd.read_parquet(path)
    missing = {"problem", "solution"} - set(df.columns)
    if missing:
        raise ValueError(f"{split}: missing columns {sorted(missing)}")
    return df


def encode_sft_example(
    tokenizer,
    *,
    row_id: int,
    problem: str,
    solution: str,
    prompt_formatter: PromptFormatter,
) -> EncodedSFTExample:
    prompt = prompt_formatter(problem)
    completion = solution.strip()
    if not completion:
        raise ValueError(f"Row {row_id}: empty solution.")

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
    target_ids = [*target_ids, tokenizer.eos_token_id]

    return EncodedSFTExample(
        row_id=row_id,
        prompt=prompt,
        completion=completion,
        input_ids=[*prompt_ids, *target_ids],
        labels=[IGNORE_INDEX] * len(prompt_ids) + target_ids.copy(),
        prompt_length=len(prompt_ids),
        target_length=len(target_ids),
    )


def _encode_split(
    data_dir: Path,
    split: str,
    *,
    tokenizer,
    prompt_formatter: PromptFormatter,
) -> list[EncodedSFTExample]:
    df = _load_split(data_dir, split)
    return [
        encode_sft_example(
            tokenizer,
            row_id=i,
            problem=str(row.problem),
            solution=str(row.solution),
            prompt_formatter=prompt_formatter,
        )
        for i, row in enumerate(df.itertuples(index=False))
    ]


def _make_collator(tokenizer) -> DataCollatorForSeq2Seq:
    return DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=IGNORE_INDEX,
        return_tensors="pt",
    )


def _assert_contract(examples: list[EncodedSFTExample], *, tokenizer) -> None:
    if not examples:
        raise ValueError("SFT dataset is empty.")

    for ex in examples:
        if len(ex.input_ids) != len(ex.labels):
            raise AssertionError(f"Row {ex.row_id}: input/label length mismatch.")
        if any(label != IGNORE_INDEX for label in ex.labels[: ex.prompt_length]):
            raise AssertionError(f"Row {ex.row_id}: prompt contributes to loss.")

        target_input = ex.input_ids[ex.prompt_length :]
        target_labels = ex.labels[ex.prompt_length :]
        if not target_labels:
            raise AssertionError(f"Row {ex.row_id}: no supervised target.")
        if target_input != target_labels:
            raise AssertionError(f"Row {ex.row_id}: target label mismatch.")
        if target_labels[-1] != tokenizer.eos_token_id:
            raise AssertionError(f"Row {ex.row_id}: EOS is not supervised.")

    batch = _make_collator(tokenizer)(
        [ex.as_record() for ex in examples[: min(8, len(examples))]]
    )
    padding = batch["attention_mask"] == 0
    if padding.any() and not torch.all(batch["labels"][padding] == IGNORE_INDEX):
        raise AssertionError("Padding contributes to loss.")


def _length_stats(values: list[int]) -> dict[str, float | int]:
    x = np.asarray(values, dtype=np.int64)
    return {
        "p50": float(np.quantile(x, 0.50)),
        "p90": float(np.quantile(x, 0.90)),
        "p95": float(np.quantile(x, 0.95)),
        "p99": float(np.quantile(x, 0.99)),
        "max": int(x.max()),
    }


def _print_lengths(name: str, values: list[int]) -> None:
    s = _length_stats(values)
    print(name)
    print(f"- p50: {s['p50']:.0f}")
    print(f"- p90: {s['p90']:.0f}")
    print(f"- p95: {s['p95']:.0f}")
    print(f"- p99: {s['p99']:.0f}")
    print(f"- max: {s['max']}")


def _print_boundary(ex: EncodedSFTExample, *, tokenizer) -> None:
    start = max(0, ex.prompt_length - 6)
    end = min(ex.full_length, ex.prompt_length + 8)
    rows = []

    for pos in range(start, end):
        token_id = ex.input_ids[pos]
        label = ex.labels[pos]
        rows.append(
            [
                pos,
                "prompt" if pos < ex.prompt_length else "target",
                token_id,
                repr(tokenizer.convert_ids_to_tokens(token_id)),
                "MASK" if label == IGNORE_INDEX else label,
            ]
        )

    print(f"Example [{ex.row_id}]")
    print("\nPrompt\n------")
    print(ex.prompt)
    print("Completion\n----------")
    print(ex.completion)
    print("\nBoundary token inspection")
    print(
        tabulate(
            rows,
            headers=["pos", "role", "input_id", "token", "label"],
            tablefmt="rounded_grid",
            disable_numparse=True,
        )
    )
    print()


def inspect_sft_data(
    *,
    model_path: Path,
    data_dir: Path,
    prompt_name: str,
    max_length: int,
    samples: int,
    seed: int,
) -> None:
    if max_length <= 0:
        raise ValueError("max_length must be positive.")
    if samples < 0:
        raise ValueError("samples cannot be negative.")

    tokenizer = load_training_tokenizer(model_path)
    prompt_formatter = get_prompt_formatter(prompt_name)
    encoded: dict[str, list[EncodedSFTExample]] = {}

    print("SFT data inspection")
    print(f"- model: {model_path}")
    print(f"- prompt: {prompt_name}")
    print(f"- max_length: {max_length}\n")

    for split in ("train", "dev"):
        examples = _encode_split(
            data_dir,
            split,
            tokenizer=tokenizer,
            prompt_formatter=prompt_formatter,
        )
        _assert_contract(examples, tokenizer=tokenizer)
        encoded[split] = examples

        overlong = sum(ex.full_length > max_length for ex in examples)
        print(f"{split.capitalize()} ({len(examples)} examples)")
        _print_lengths("Prompt tokens", [ex.prompt_length for ex in examples])
        _print_lengths(
            "Target tokens (solution + EOS)",
            [ex.target_length for ex in examples],
        )
        _print_lengths("Full sequence tokens", [ex.full_length for ex in examples])
        print(
            f"Full sequence > {max_length}: "
            f"{overlong} ({100 * overlong / len(examples):.2f}%)"
        )
        print("Prompt positions supervised: 0 [PASS]")
        print("Target labels == target input_ids: [PASS]")
        print("EOS supervised on all examples: [PASS]")
        print("Padding positions supervised: 0 [PASS]\n")

    if samples:
        chosen = random.Random(seed).sample(
            encoded["train"],
            min(samples, len(encoded["train"])),
        )
        print("Human inspection samples\n")
        for ex in chosen:
            _print_boundary(ex, tokenizer=tokenizer)


def _prepare_examples(
    *,
    data_dir: Path,
    split: str,
    tokenizer,
    prompt_formatter: PromptFormatter,
    max_length: int,
    overlong_policy: str,
) -> list[EncodedSFTExample]:
    if overlong_policy not in {"error", "drop"}:
        raise ValueError("overlong_policy must be 'error' or 'drop'.")

    examples = _encode_split(
        data_dir,
        split,
        tokenizer=tokenizer,
        prompt_formatter=prompt_formatter,
    )
    _assert_contract(examples, tokenizer=tokenizer)

    overlong = [ex for ex in examples if ex.full_length > max_length]
    if overlong and overlong_policy == "error":
        raise ValueError(
            f"{split}: {len(overlong)} examples exceed max_length={max_length}; "
            f"longest={max(ex.full_length for ex in overlong)}. "
            "Increase --max-length or explicitly use --overlong-policy drop."
        )
    if overlong_policy == "drop":
        examples = [ex for ex in examples if ex.full_length <= max_length]
    if not examples:
        raise ValueError(f"{split}: no examples remain.")
    return examples


# ---------------------------------------------------------------------------
# Runtime helpers
# ---------------------------------------------------------------------------


def _resolve_precision(precision: str) -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for SFT.")
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


def _load_model(model_path: Path, *, precision: str):
    if not model_path.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=_dtype_for_precision(precision),
    )
    model.config.use_cache = False
    return model


def build_lora_config(
    *,
    r: int,
    alpha: int,
    dropout: float,
    target_modules: str,
) -> LoraConfig:
    if r <= 0 or alpha <= 0:
        raise ValueError("LoRA rank and alpha must be positive.")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("LoRA dropout must be in [0, 1).")
    if not target_modules.strip():
        raise ValueError("LoRA target modules cannot be empty.")

    targets: str | list[str]
    if target_modules == "all-linear":
        targets = "all-linear"
    else:
        targets = [
            item.strip()
            for item in target_modules.split(",")
            if item.strip()
        ]
        if not targets:
            raise ValueError("LoRA target modules cannot be empty.")

    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=targets,
        bias="none",
        task_type="CAUSAL_LM",
    )


def _trainable_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _gpu_memory() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {}
    try:
        free, total = torch.cuda.mem_get_info()
        return {
            "gpu": torch.cuda.get_device_name(0),
            "total_gib": total / GIB,
            "free_gib": free / GIB,
            "allocated_gib": torch.cuda.memory_allocated() / GIB,
            "reserved_gib": torch.cuda.memory_reserved() / GIB,
            "peak_gib": torch.cuda.max_memory_allocated() / GIB,
        }
    # GPU telemetry is best-effort and must not fail a training run.
    except Exception:  # noqa: BLE001
        return {"gpu": torch.cuda.get_device_name(0)}


def _print_gpu(title: str) -> None:
    m = _gpu_memory()
    if not m:
        return
    print(title)
    print(f"- GPU: {m['gpu']}")
    for key, label in (
        ("total_gib", "total"),
        ("free_gib", "free"),
        ("allocated_gib", "allocated"),
        ("reserved_gib", "reserved"),
        ("peak_gib", "peak allocated"),
    ):
        if key in m:
            print(f"- {label}: {m[key]:.2f} GiB")


def _is_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower()


def _oom(
    exc: BaseException,
    *,
    stage: str,
    batch_size: int,
    eval_batch_size: int | None,
    gradient_accumulation: int,
    max_length: int,
    precision: str,
    gradient_checkpointing: bool,
) -> None:
    print("\nCUDA OUT OF MEMORY", file=sys.stderr)
    print(f"- stage: {stage}", file=sys.stderr)
    print(f"- batch_size: {batch_size}", file=sys.stderr)
    if eval_batch_size is not None:
        print(f"- eval_batch_size: {eval_batch_size}", file=sys.stderr)
    print(f"- gradient_accumulation: {gradient_accumulation}", file=sys.stderr)
    print(f"- max_length: {max_length}", file=sys.stderr)
    print(f"- precision: {precision}", file=sys.stderr)
    print(f"- gradient_checkpointing: {gradient_checkpointing}", file=sys.stderr)

    m = _gpu_memory()
    if "total_gib" in m:
        print(f"- GPU: {m['gpu']}", file=sys.stderr)
        print(
            "- VRAM total/free/allocated/reserved/peak: "
            f"{m['total_gib']:.2f} / {m['free_gib']:.2f} / "
            f"{m['allocated_gib']:.2f} / {m['reserved_gib']:.2f} / "
            f"{m['peak_gib']:.2f} GiB",
            file=sys.stderr,
        )

    print("\nStandard knobs to investigate:", file=sys.stderr)
    print("- reduce --batch-size", file=sys.stderr)
    print("- reduce --max-length", file=sys.stderr)
    print("- enable --gradient-checkpointing", file=sys.stderr)
    print(
        "- increase --gradient-accumulation if preserving effective batch size",
        file=sys.stderr,
    )
    raise RuntimeError("CUDA out of memory; diagnostics printed above.") from exc


def _to_dataset(examples: list[EncodedSFTExample]) -> Dataset:
    return Dataset.from_list([ex.as_record() for ex in examples])


def _token_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = logits[:, :-1].argmax(dim=-1)
    targets = labels[:, 1:]
    mask = targets != IGNORE_INDEX
    if not mask.any():
        return 0.0
    return float((predictions[mask] == targets[mask]).float().mean().item())


def _plot(
    x: list[float],
    y: list[float],
    *,
    ylabel: str,
    title: str,
    path: Path,
) -> None:
    if not x:
        return
    fig = plt.figure()
    plt.plot(x, y)
    plt.xlabel("optimizer step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# One-batch overfit sanity
# ---------------------------------------------------------------------------


def overfit_one_batch(
    *,
    model_path: Path,
    data_dir: Path,
    prompt_name: str,
    max_length: int,
    overlong_policy: str,
    batch_size: int,
    steps: int,
    learning_rate: float,
    max_grad_norm: float,
    precision: str,
    peft_method: str,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_target_modules: str,
    seed: int,
    output_dir: Path,
) -> None:
    if batch_size <= 0 or steps <= 0 or learning_rate <= 0:
        raise ValueError("batch_size, steps and learning_rate must be positive.")
    if peft_method not in {"none", "lora"}:
        raise ValueError("peft_method must be 'none' or 'lora'.")

    precision = _resolve_precision(precision)
    random.seed(seed)
    torch.manual_seed(seed)

    tokenizer = load_training_tokenizer(model_path)
    examples = _prepare_examples(
        data_dir=data_dir,
        split="train",
        tokenizer=tokenizer,
        prompt_formatter=get_prompt_formatter(prompt_name),
        max_length=max_length,
        overlong_policy=overlong_policy,
    )
    chosen = random.Random(seed).sample(examples, min(batch_size, len(examples)))
    batch = _make_collator(tokenizer)([ex.as_record() for ex in chosen])
    batch = {key: value.to("cuda") for key, value in batch.items()}

    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "overfit_log.jsonl"
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = optimizer = None
    try:
        model = _load_model(model_path, precision=precision)
        if peft_method == "lora":
            model = get_peft_model(
                model,
                build_lora_config(
                    r=lora_r,
                    alpha=lora_alpha,
                    dropout=lora_dropout,
                    target_modules=lora_target_modules,
                ),
            )
        model = model.to("cuda")
        model.train()
        optimizer = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=learning_rate,
        )
        scaler = torch.amp.GradScaler("cuda", enabled=(precision == "fp16"))
        amp = precision in {"bf16", "fp16"}
        amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16

        print("One-batch overfit sanity")
        print(f"- examples: {len(chosen)}")
        print(f"- steps: {steps}")
        print(f"- learning_rate: {learning_rate}")
        print(f"- precision: {precision}")
        print(f"- PEFT: {peft_method}")
        if peft_method == "lora":
            print(f"- LoRA r/alpha/dropout: {lora_r} / {lora_alpha} / {lora_dropout}")
            print(f"- LoRA target modules: {lora_target_modules}")
        print(f"- gradient dimension: {_trainable_params(model):,}\n")

        records = []
        with log_path.open("w", encoding="utf-8") as file:
            for step in range(1, steps + 1):
                optimizer.zero_grad(set_to_none=True)

                with torch.autocast("cuda", dtype=amp_dtype, enabled=amp):
                    outputs = model(**batch)
                    loss = outputs.loss

                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at step {step}.")

                token_acc = _token_accuracy(outputs.logits.detach(), batch["labels"])
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)

                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=max_grad_norm,
                    )
                )
                if not math.isfinite(grad_norm):
                    raise FloatingPointError(
                        f"Non-finite gradient norm at step {step}."
                    )

                scaler.step(optimizer)
                scaler.update()

                record = {
                    "step": step,
                    "loss": float(loss.detach()),
                    "mean_token_accuracy": token_acc,
                    "grad_norm": grad_norm,
                    "grad_above_clip_threshold": grad_norm > max_grad_norm,
                    "gpu_allocated_gib": torch.cuda.memory_allocated() / GIB,
                    "gpu_peak_gib": torch.cuda.max_memory_allocated() / GIB,
                }
                records.append(record)
                file.write(json.dumps(record) + "\n")
                file.flush()

                print(
                    f"[{step:03d}/{steps}] loss={record['loss']:.4f} | "
                    f"token_acc={token_acc:.3f} | grad_norm={grad_norm:.3f} | "
                    f"clip={'Y' if grad_norm > max_grad_norm else 'N'}",
                    flush=True,
                )

        step_x = [float(r["step"]) for r in records]
        _plot(
            step_x,
            [float(r["loss"]) for r in records],
            ylabel="loss",
            title="One-batch overfit loss",
            path=output_dir / "overfit_loss.png",
        )
        _plot(
            step_x,
            [float(r["grad_norm"]) for r in records],
            ylabel="gradient L2 norm",
            title="One-batch gradient norm",
            path=output_dir / "overfit_grad_norm.png",
        )

        print("\nOverfit summary")
        print(f"- initial loss: {records[0]['loss']:.4f}")
        print(f"- final loss: {records[-1]['loss']:.4f}")
        print(f"- final token accuracy: {records[-1]['mean_token_accuracy']:.3f}")
        _print_gpu("GPU memory")
        print("\nArtifacts")
        print(f"- {log_path}")
        print(f"- {output_dir / 'overfit_loss.png'}")
        print(f"- {output_dir / 'overfit_grad_norm.png'}")

    except BaseException as exc:
        if _is_oom(exc):
            _oom(
                exc,
                stage="one-batch overfit",
                batch_size=batch_size,
                eval_batch_size=None,
                gradient_accumulation=1,
                max_length=max_length,
                precision=precision,
                gradient_checkpointing=False,
            )
        raise
    finally:
        del optimizer
        del model
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Full SFT
# ---------------------------------------------------------------------------


class TrainingLogCallback(TrainerCallback):
    def __init__(self, log_path: Path, max_grad_norm: float) -> None:
        self.log_path = log_path
        self.max_grad_norm = max_grad_norm
        self.grad_points = 0
        self.clipped_points = 0
        self.log_path.write_text("", encoding="utf-8")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return control

        record: dict[str, Any] = {
            "step": int(state.global_step),
            "epoch": float(state.epoch) if state.epoch is not None else None,
            **logs,
        }

        for key in ("loss", "eval_loss", "grad_norm"):
            if key in record and not math.isfinite(float(record[key])):
                raise FloatingPointError(
                    f"Non-finite {key} at step {state.global_step}."
                )

        grad_norm = record.get("grad_norm")
        if grad_norm is not None:
            grad_norm = float(grad_norm)
            record["grad_above_clip_threshold"] = grad_norm > self.max_grad_norm
            self.grad_points += 1
            self.clipped_points += int(grad_norm > self.max_grad_norm)

        if torch.cuda.is_available():
            record["gpu_allocated_gib"] = torch.cuda.memory_allocated() / GIB
            record["gpu_reserved_gib"] = torch.cuda.memory_reserved() / GIB
            record["gpu_peak_gib"] = torch.cuda.max_memory_allocated() / GIB

        with self.log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

        fields = [f"step={state.global_step}"]
        for key, label, fmt in (
            ("loss", "loss", ".4f"),
            ("eval_loss", "eval_loss", ".4f"),
            ("mean_token_accuracy", "token_acc", ".3f"),
            ("entropy", "entropy", ".3f"),
            ("grad_norm", "grad_norm", ".3f"),
            ("learning_rate", "lr", ".3e"),
        ):
            if key in record:
                fields.append(f"{label}={format(float(record[key]), fmt)}")

        if grad_norm is not None:
            fields.append(f"clip={'Y' if grad_norm > self.max_grad_norm else 'N'}")
        if "num_tokens" in record:
            fields.append(f"tokens={int(float(record['num_tokens'])):,}")
        if "gpu_allocated_gib" in record:
            fields.append(f"vram={record['gpu_allocated_gib']:.2f}GiB")

        print(" | ".join(fields), flush=True)
        return control

    def clipping_summary(self) -> dict[str, Any]:
        return {
            "logged_grad_points": self.grad_points,
            "logged_points_above_clip": self.clipped_points,
            "logged_clip_ratio": (
                self.clipped_points / self.grad_points if self.grad_points else 0.0
            ),
        }


def _plot_history(history: list[dict[str, Any]], output_dir: Path) -> None:
    train = [r for r in history if "loss" in r and "eval_loss" not in r]
    dev = [r for r in history if "eval_loss" in r]

    if train or dev:
        fig = plt.figure()
        if train:
            plt.plot(
                [r["step"] for r in train],
                [r["loss"] for r in train],
                label="train loss",
            )
        if dev:
            plt.plot(
                [r["step"] for r in dev],
                [r["eval_loss"] for r in dev],
                marker="o",
                label="dev loss",
            )
        plt.xlabel("optimizer step")
        plt.ylabel("loss")
        plt.title("SFT train / dev loss")
        plt.legend()
        plt.tight_layout()
        fig.savefig(output_dir / "loss_curve.png", dpi=160)
        plt.close(fig)

    for metric, filename, ylabel in (
        ("grad_norm", "grad_norm_curve.png", "gradient L2 norm"),
        ("learning_rate", "learning_rate_curve.png", "learning rate"),
        ("mean_token_accuracy", "token_accuracy_curve.png", "mean token accuracy"),
    ):
        rows = [r for r in history if metric in r]
        _plot(
            [float(r["step"]) for r in rows],
            [float(r[metric]) for r in rows],
            ylabel=ylabel,
            title=metric,
            path=output_dir / filename,
        )


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


def build_sft_config(*, warmup_ratio: float, **kwargs: Any) -> SFTConfig:
    if not 0.0 <= warmup_ratio <= 1.0:
        raise ValueError("warmup_ratio must be between 0 and 1.")

    # Transformers 5 uses a fractional warmup_steps value as the ratio.
    return SFTConfig(warmup_steps=warmup_ratio, **kwargs)


def train_sft(
    *,
    model_path: Path,
    data_dir: Path,
    prompt_name: str,
    max_length: int,
    overlong_policy: str,
    output_dir: Path,
    epochs: float,
    learning_rate: float,
    batch_size: int,
    eval_batch_size: int,
    gradient_accumulation: int,
    weight_decay: float,
    warmup_ratio: float,
    scheduler: str,
    max_grad_norm: float,
    optim: str,
    precision: str,
    peft_method: str,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_target_modules: str,
    gradient_checkpointing: bool,
    logging_steps: int,
    eval_steps: int,
    save_steps: int,
    save_total_limit: int,
    seed: int,
    resume_from_checkpoint: Path | None,
) -> None:
    if min(batch_size, eval_batch_size, gradient_accumulation) <= 0:
        raise ValueError("batch sizes and gradient_accumulation must be positive.")
    if min(logging_steps, eval_steps, save_steps) <= 0:
        raise ValueError("logging/eval/save steps must be positive.")
    if epochs <= 0 or learning_rate <= 0:
        raise ValueError("epochs and learning_rate must be positive.")
    if peft_method not in {"none", "lora"}:
        raise ValueError("peft_method must be 'none' or 'lora'.")

    precision = _resolve_precision(precision)
    random.seed(seed)
    torch.manual_seed(seed)

    tokenizer = load_training_tokenizer(model_path)
    formatter = get_prompt_formatter(prompt_name)
    train_examples = _prepare_examples(
        data_dir=data_dir,
        split="train",
        tokenizer=tokenizer,
        prompt_formatter=formatter,
        max_length=max_length,
        overlong_policy=overlong_policy,
    )
    dev_examples = _prepare_examples(
        data_dir=data_dir,
        split="dev",
        tokenizer=tokenizer,
        prompt_formatter=formatter,
        max_length=max_length,
        overlong_policy=overlong_policy,
    )

    train_dataset = _to_dataset(train_examples)
    dev_dataset = _to_dataset(dev_examples)
    collator = _make_collator(tokenizer)

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = trainer = None
    try:
        model = _load_model(model_path, precision=precision)
        lora_config = None
        if peft_method == "lora":
            lora_config = build_lora_config(
                r=lora_r,
                alpha=lora_alpha,
                dropout=lora_dropout,
                target_modules=lora_target_modules,
            )
            model = get_peft_model(model, lora_config)

        trainable = _trainable_params(model)
        total = sum(p.numel() for p in model.parameters())
        effective_batch = batch_size * gradient_accumulation
        estimated_steps = math.ceil(
            math.ceil(len(train_dataset) / effective_batch) * epochs
        )

        print("SFT training plan")
        print(f"- model: {model_path}")
        print(f"- prompt: {prompt_name}")
        print(f"- train/dev examples: {len(train_dataset)} / {len(dev_dataset)}")
        print(f"- max_length: {max_length}")
        print(f"- train/eval batch size: {batch_size} / {eval_batch_size}")
        print(f"- gradient accumulation: {gradient_accumulation}")
        print(f"- effective batch size: {effective_batch}")
        print(f"- epochs: {epochs}")
        print(f"- estimated optimizer steps: {estimated_steps}")
        print(f"- optimizer: {optim}")
        print(f"- learning rate: {learning_rate}")
        print(f"- scheduler: {scheduler}")
        print(f"- warmup ratio: {warmup_ratio}")
        print(f"- max grad norm: {max_grad_norm}")
        print(f"- precision: {precision}")
        print(f"- PEFT: {peft_method}")
        if peft_method == "lora":
            print(f"- LoRA r/alpha/dropout: {lora_r} / {lora_alpha} / {lora_dropout}")
            print(f"- LoRA target modules: {lora_target_modules}")
        print(f"- gradient checkpointing: {gradient_checkpointing}")
        print(f"- total parameters: {total:,}")
        print(f"- trainable parameters: {trainable:,}")
        print(f"- gradient dimension: {trainable:,}\n")
        _print_gpu("GPU before Trainer")

        run_config = {
            "git_commit": _git_commit(),
            "model": str(model_path),
            "prompt": prompt_name,
            "data_dir": str(data_dir),
            "max_length": max_length,
            "overlong_policy": overlong_policy,
            "train_examples": len(train_dataset),
            "dev_examples": len(dev_dataset),
            "epochs": epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "eval_batch_size": eval_batch_size,
            "gradient_accumulation": gradient_accumulation,
            "effective_batch_size": effective_batch,
            "weight_decay": weight_decay,
            "warmup_ratio": warmup_ratio,
            "scheduler": scheduler,
            "max_grad_norm": max_grad_norm,
            "optimizer": optim,
            "precision": precision,
            "peft_method": peft_method,
            "lora": (
                {
                    "r": lora_r,
                    "alpha": lora_alpha,
                    "dropout": lora_dropout,
                    "target_modules": lora_target_modules,
                }
                if peft_method == "lora"
                else None
            ),
            "gradient_checkpointing": gradient_checkpointing,
            "logging_steps": logging_steps,
            "eval_steps": eval_steps,
            "save_steps": save_steps,
            "save_total_limit": save_total_limit,
            "seed": seed,
            "torch": torch.__version__,
            "transformers": version("transformers"),
            "trl": version("trl"),
            "peft": version("peft"),
            "datasets": version("datasets"),
        }
        (output_dir / "run_config.json").write_text(
            json.dumps(run_config, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        callback = TrainingLogCallback(
            output_dir / "train_log.jsonl",
            max_grad_norm=max_grad_norm,
        )

        args = build_sft_config(
            warmup_ratio=warmup_ratio,
            output_dir=str(output_dir),
            num_train_epochs=epochs,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation,
            weight_decay=weight_decay,
            lr_scheduler_type=scheduler,
            max_grad_norm=max_grad_norm,
            optim=optim,
            bf16=(precision == "bf16"),
            fp16=(precision == "fp16"),
            gradient_checkpointing=gradient_checkpointing,
            logging_strategy="steps",
            logging_steps=logging_steps,
            logging_first_step=True,
            logging_nan_inf_filter=False,
            eval_strategy="steps",
            eval_steps=eval_steps,
            eval_on_start=True,
            save_strategy="steps",
            save_steps=save_steps,
            save_total_limit=save_total_limit,
            report_to="none",
            seed=seed,
            data_seed=seed,
            disable_tqdm=True,
            dataloader_num_workers=0,
            packing=False,
            padding_free=False,
            loss_type="nll",
            use_cache=False,
            max_length=None,
            dataset_kwargs={"skip_prepare_dataset": True},
        )

        trainer = SFTTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            processing_class=tokenizer,
            data_collator=collator,
            callbacks=[callback],
        )

        print("\nStarting SFT\n")
        train_result = trainer.train(
            resume_from_checkpoint=(
                str(resume_from_checkpoint)
                if resume_from_checkpoint is not None
                else None
            )
        )

        print("\nRunning final dev-loss evaluation...")
        final_eval = trainer.evaluate()

        final_model_dir = output_dir / "final-model"
        trainer.save_model(str(final_model_dir))
        tokenizer.save_pretrained(final_model_dir)
        _plot_history(trainer.state.log_history, output_dir)

        summary = {
            "train_metrics": train_result.metrics,
            "final_eval": final_eval,
            "clipping": callback.clipping_summary(),
            "gpu": _gpu_memory(),
            "final_model": str(final_model_dir),
        }
        (output_dir / "train_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )

        clipping = callback.clipping_summary()
        print("\nSFT complete")
        print(f"- final dev loss: {float(final_eval['eval_loss']):.4f}")
        print(
            "- logged grad points above clip threshold: "
            f"{clipping['logged_points_above_clip']} / "
            f"{clipping['logged_grad_points']}"
        )
        _print_gpu("GPU after training")

        print("\nArtifacts")
        for path in (
            output_dir / "run_config.json",
            output_dir / "train_log.jsonl",
            output_dir / "loss_curve.png",
            output_dir / "grad_norm_curve.png",
            output_dir / "learning_rate_curve.png",
            output_dir / "token_accuracy_curve.png",
            output_dir / "train_summary.json",
            final_model_dir,
        ):
            if path.exists():
                print(f"- {path}")

    except BaseException as exc:
        if _is_oom(exc):
            _oom(
                exc,
                stage="full SFT",
                batch_size=batch_size,
                eval_batch_size=eval_batch_size,
                gradient_accumulation=gradient_accumulation,
                max_length=max_length,
                precision=precision,
                gradient_checkpointing=gradient_checkpointing,
            )
        raise
    finally:
        del trainer
        del model
        torch.cuda.empty_cache()
