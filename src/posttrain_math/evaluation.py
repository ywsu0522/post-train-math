from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tabulate import tabulate

from posttrain_math.answers import extract_last_boxed, verify_boxed_answers
from posttrain_math.distributed import (
    barrier,
    destroy_process_group_if_owned,
    get_distributed_context,
    init_process_group_if_needed,
)
from posttrain_math.modeling import TextGenerator
from posttrain_math.prompting import PromptFormatter


def _normalize_optional_string(value: Any) -> str | None:
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    value = str(value).strip()
    return value or None


def load_eval_split(data_dir: Path, split: str) -> pd.DataFrame:
    if split not in {"dev", "test"}:
        raise ValueError("Evaluation split must be 'dev' or 'test'")

    path = data_dir / f"{split}.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"Processed split not found: {path}")

    df = pd.read_parquet(path)
    required = {
        "problem",
        "solution",
        "type",
        "level",
        "gt_boxed",
        "eval_eligible",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "Evaluation data missing columns: "
            f"{sorted(missing)}. Re-run `posttrain-math data prepare`."
        )

    return df


def _length_statistics(lengths: list[int]) -> dict[str, float | int]:
    values = np.asarray(lengths, dtype=np.int64)
    return {
        "median": float(np.median(values)),
        "max": int(values.max()),
    }


def _print_token_diagnostics(
    *,
    prompt_lengths: list[int],
    solution_lengths: list[int],
    max_new_tokens: int,
) -> dict[str, Any]:
    prompt_stats = _length_statistics(prompt_lengths)
    solution_stats = _length_statistics(solution_lengths)

    over_budget = sum(length > max_new_tokens for length in solution_lengths)
    ratio = over_budget / len(solution_lengths) if solution_lengths else 0.0

    print("Token diagnostics")
    print(
        "  prompt tokens: "
        f"median={prompt_stats['median']:.0f}, max={prompt_stats['max']}"
    )
    print(
        "  gold solution tokens: "
        f"median={solution_stats['median']:.0f}, max={solution_stats['max']}"
    )
    print(
        "  gold solution > max_new_tokens "
        f"({max_new_tokens}): {over_budget} ({100.0 * ratio:.2f}%)"
    )

    return {
        "prompt_tokens": prompt_stats,
        "gold_solution_tokens": solution_stats,
        "max_new_tokens": max_new_tokens,
        "gold_solution_over_budget": {
            "count": over_budget,
            "ratio": ratio,
        },
    }


def _update_group(
    groups: dict[str, dict[str, int]],
    key: str,
    *,
    correct: bool,
) -> None:
    bucket = groups.setdefault(key, {"correct": 0, "n": 0})
    bucket["n"] += 1
    bucket["correct"] += int(correct)


def _finalize_groups(
    groups: dict[str, dict[str, int]],
) -> dict[str, dict[str, int | float]]:
    return {
        key: {
            "correct": values["correct"],
            "n": values["n"],
            "accuracy": values["correct"] / values["n"] if values["n"] else 0.0,
        }
        for key, values in sorted(groups.items())
    }


def _print_group_accuracy(
    title: str,
    groups: dict[str, dict[str, int | float]],
) -> None:
    rows = [
        [
            key,
            f"{values['correct']} / {values['n']}",
            f"{float(values['accuracy']):.2%}",
        ]
        for key, values in groups.items()
    ]

    print()
    print(title)
    print(
        tabulate(
            rows,
            headers=["Group", "Correct / N", "Accuracy"],
            tablefmt="rounded_grid",
            disable_numparse=True,
        )
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def _metrics_from_records(
    records: list[dict[str, Any]],
    *,
    split: str,
    prompt_name: str,
    generator_metadata: dict[str, Any],
    generation_config: dict[str, Any],
    source_rows: int,
    eligible_rows: int,
    excluded_rows: int,
    limit: int | None,
    token_diagnostics: dict[str, Any],
    world_size: int,
    per_device_batch_size: int | None,
) -> dict[str, Any]:
    counters = {
        "num_examples": 0,
        "num_gt_boxed": 0,
        "num_gt_parseable": 0,
        "num_pred_boxed": 0,
        "num_pred_parseable": 0,
        "num_correct": 0,
    }
    by_level: dict[str, dict[str, int]] = {}
    by_type: dict[str, dict[str, int]] = {}

    for record in records:
        counters["num_examples"] += 1
        counters["num_gt_boxed"] += int(record["gt_boxed"] is not None)
        counters["num_gt_parseable"] += int(bool(record["gold_parseable"]))
        counters["num_pred_boxed"] += int(record["pred_boxed"] is not None)
        counters["num_pred_parseable"] += int(bool(record["prediction_parseable"]))
        counters["num_correct"] += int(bool(record["correct"]))

        _update_group(
            by_level,
            str(record["level"]),
            correct=bool(record["correct"]),
        )
        _update_group(
            by_type,
            str(record["type"]),
            correct=bool(record["correct"]),
        )

    num_examples = counters["num_examples"]
    accuracy = counters["num_correct"] / num_examples if num_examples else 0.0
    boxed_output_rate = (
        counters["num_pred_boxed"] / num_examples if num_examples else 0.0
    )
    parseable_output_rate = (
        counters["num_pred_parseable"] / num_examples if num_examples else 0.0
    )
    gt_boxed_coverage = (
        counters["num_gt_boxed"] / num_examples if num_examples else 0.0
    )

    return {
        "split": split,
        "prompt_strategy": prompt_name,
        "generator": generator_metadata,
        "distributed": {
            "world_size": world_size,
            "per_device_batch_size": per_device_batch_size,
        },
        "generation_config": generation_config,
        "cohort": {
            "source_rows": source_rows,
            "eligible_rows": eligible_rows,
            "excluded_rows": excluded_rows,
            "evaluated_rows": num_examples,
            "limit": limit,
        },
        "token_diagnostics": token_diagnostics,
        **counters,
        "gt_boxed_coverage": gt_boxed_coverage,
        "boxed_output_rate": boxed_output_rate,
        "parseable_output_rate": parseable_output_rate,
        "accuracy": accuracy,
        "accuracy_by_level": _finalize_groups(by_level),
        "accuracy_by_type": _finalize_groups(by_type),
    }


def _print_final_metrics(
    metrics: dict[str, Any],
    *,
    predictions_path: Path,
    metrics_path: Path,
) -> None:
    cohort = metrics["cohort"]
    print()
    print("Evaluation")
    print(f"  split:                 {metrics['split']}")
    print(f"  prompt strategy:       {metrics['prompt_strategy']}")
    print(
        "  eligible cohort:       "
        f"{cohort['eligible_rows']} / {cohort['source_rows']}"
    )
    print(f"  evaluated examples:    {metrics['num_examples']}")
    print(
        "  correct:               "
        f"{metrics['num_correct']} / {metrics['num_examples']}"
    )
    print(f"  accuracy:              {metrics['accuracy']:.2%}")
    print(f"  boxed output rate:     {metrics['boxed_output_rate']:.2%}")
    print(
        "  parseable output rate: "
        f"{metrics['parseable_output_rate']:.2%}"
    )
    print(f"  inference world size:  {metrics['distributed']['world_size']}")

    _print_group_accuracy(
        "Accuracy by level",
        metrics["accuracy_by_level"],
    )
    _print_group_accuracy(
        "Accuracy by type",
        metrics["accuracy_by_type"],
    )

    print()
    print(f"  predictions: {predictions_path}")
    print(f"  metrics:     {metrics_path}")


def evaluate(
    generator: TextGenerator,
    *,
    prompt_formatter: PromptFormatter,
    prompt_name: str,
    data_dir: Path,
    split: str,
    output_dir: Path,
    max_new_tokens: int,
    limit: int | None = None,
) -> dict[str, Any]:
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")

    context = get_distributed_context(require_cuda=False)
    owns_process_group = init_process_group_if_needed(context)

    try:
        source_df = load_eval_split(data_dir, split)
        source_rows = len(source_df)

        eligible_df = source_df[source_df["eval_eligible"].astype(bool)].copy()
        eligible_rows = len(eligible_df)
        excluded_rows = source_rows - eligible_rows

        if eligible_rows == 0:
            raise ValueError(f"{split}: evaluation cohort is empty")

        if limit is not None:
            if limit <= 0:
                raise ValueError("limit must be positive")
            df = eligible_df.head(limit).copy()
        else:
            df = eligible_df.copy()

        df = df.reset_index(drop=True)
        df["eval_index"] = np.arange(len(df), dtype=np.int64)
        num_examples = len(df)

        output_dir.mkdir(parents=True, exist_ok=True)
        predictions_path = output_dir / "predictions.jsonl"
        metrics_path = output_dir / "metrics.json"

        token_diagnostics: dict[str, Any] = {}
        if context.is_main_process:
            prompts = [
                prompt_formatter(str(problem))
                for problem in df["problem"]
            ]
            solutions = [str(solution) for solution in df["solution"]]
            token_diagnostics = _print_token_diagnostics(
                prompt_lengths=generator.count_tokens(prompts),
                solution_lengths=generator.count_tokens(solutions),
                max_new_tokens=max_new_tokens,
            )

            print()
            print("Evaluation cohort")
            print(f"  source rows:   {source_rows}")
            print(f"  eligible rows: {eligible_rows}")
            print(f"  excluded rows: {excluded_rows}")
            if limit is not None:
                print(f"  evaluated rows: {num_examples} (--limit {limit})")
            else:
                print(f"  evaluated rows: {num_examples}")
            print(f"  inference processes: {context.world_size}")
            print()

        local_df = df.iloc[context.rank :: context.world_size].copy()
        local_prompts = [
            prompt_formatter(str(problem))
            for problem in local_df["problem"]
        ]

        if context.is_distributed:
            shard_dir = output_dir / "shards"
            shard_dir.mkdir(parents=True, exist_ok=True)
            local_predictions_path = (
                shard_dir / f"predictions.rank{context.rank:03d}.jsonl"
            )
        else:
            local_predictions_path = predictions_path

        print(
            f"[rank {context.rank}] generating {len(local_df)} / "
            f"{num_examples} examples on {context.device}",
            flush=True,
        )

        generations = generator.iter_generate(
            local_prompts,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

        written = 0
        with local_predictions_path.open("w", encoding="utf-8") as file:
            for (_, row), prompt, generation in zip(
                local_df.iterrows(),
                local_prompts,
                generations,
                strict=True,
            ):
                gt_boxed = _normalize_optional_string(row["gt_boxed"])
                pred_boxed = extract_last_boxed(generation)

                correct, gold_parseable, prediction_parseable = (
                    verify_boxed_answers(
                        gt_boxed,
                        pred_boxed,
                    )
                )

                if not gold_parseable:
                    raise RuntimeError(
                        "Prepared evaluation cohort contains a gold answer that is "
                        "no longer parseable. Re-run `posttrain-math data prepare` "
                        "and verify the locked math-verify version."
                    )

                record = {
                    "eval_index": int(row["eval_index"]),
                    "problem": str(row["problem"]),
                    "type": str(row["type"]),
                    "level": str(row["level"]),
                    "prompt_strategy": prompt_name,
                    "prompt": prompt,
                    "gt_boxed": gt_boxed,
                    "generation": generation,
                    "pred_boxed": pred_boxed,
                    "gold_parseable": gold_parseable,
                    "prediction_parseable": prediction_parseable,
                    "correct": correct,
                }

                file.write(json.dumps(record, ensure_ascii=False) + "\n")
                file.flush()
                written += 1

                status = "CORRECT" if correct else "INCORRECT"
                gt_display = gt_boxed if gt_boxed is not None else "<none>"
                pred_display = (
                    pred_boxed if pred_boxed is not None else "<none>"
                )
                global_number = int(row["eval_index"]) + 1
                print(
                    f"[rank {context.rank}] "
                    f"[{global_number}/{num_examples}] {status} | "
                    f"gt={gt_display} | pred={pred_display}",
                    flush=True,
                )

        if written != len(local_df):
            raise RuntimeError(
                f"Rank {context.rank}: generation count mismatch: "
                f"expected {len(local_df)}, got {written}"
            )

        barrier(context)

        metrics: dict[str, Any] = {}

        if context.is_main_process:
            if context.is_distributed:
                records: list[dict[str, Any]] = []
                shard_dir = output_dir / "shards"
                for rank in range(context.world_size):
                    shard_path = (
                        shard_dir / f"predictions.rank{rank:03d}.jsonl"
                    )
                    if not shard_path.is_file():
                        raise RuntimeError(
                            f"Missing evaluation shard: {shard_path}"
                        )
                    records.extend(_read_jsonl(shard_path))
                records.sort(key=lambda record: int(record["eval_index"]))
                _write_jsonl(predictions_path, records)
            else:
                records = _read_jsonl(predictions_path)

            actual_indices = [
                int(record["eval_index"])
                for record in records
            ]
            expected_indices = list(range(num_examples))
            if actual_indices != expected_indices:
                raise RuntimeError(
                    "Merged evaluation shards have missing, duplicate, "
                    "or out-of-order eval_index values."
                )

            generator_metadata = generator.metadata()
            per_device_batch_size = generator_metadata.get("batch_size")
            metrics = _metrics_from_records(
                records,
                split=split,
                prompt_name=prompt_name,
                generator_metadata=generator_metadata,
                generation_config={
                    "do_sample": False,
                    "max_new_tokens": max_new_tokens,
                },
                source_rows=source_rows,
                eligible_rows=eligible_rows,
                excluded_rows=excluded_rows,
                limit=limit,
                token_diagnostics=token_diagnostics,
                world_size=context.world_size,
                per_device_batch_size=(
                    int(per_device_batch_size)
                    if per_device_batch_size is not None
                    else None
                ),
            )

            metrics_path.write_text(
                json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            _print_final_metrics(
                metrics,
                predictions_path=predictions_path,
                metrics_path=metrics_path,
            )

        barrier(context)
        return metrics

    finally:
        destroy_process_group_if_owned(owns_process_group)
