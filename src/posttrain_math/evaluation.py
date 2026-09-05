from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tabulate import tabulate

from posttrain_math.answers import extract_last_boxed, verify_boxed_answers
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
    print(f"  prompt tokens: median={prompt_stats['median']:.0f}, max={prompt_stats['max']}")
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
        df = eligible_df

    prompts = [prompt_formatter(str(problem)) for problem in df["problem"]]
    solutions = [str(solution) for solution in df["solution"]]

    token_diagnostics = _print_token_diagnostics(
        prompt_lengths=generator.count_tokens(prompts),
        solution_lengths=generator.count_tokens(solutions),
        max_new_tokens=max_new_tokens,
    )

    num_examples = len(df)

    print()
    print("Evaluation cohort")
    print(f"  source rows:   {source_rows}")
    print(f"  eligible rows: {eligible_rows}")
    print(f"  excluded rows: {excluded_rows}")
    if limit is not None:
        print(f"  evaluated rows: {num_examples} (--limit {limit})")
    else:
        print(f"  evaluated rows: {num_examples}")

    print()
    print(f"Generating {num_examples} examples...")
    print()

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    metrics_path = output_dir / "metrics.json"

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

    generations = generator.iter_generate(
        prompts,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    with predictions_path.open("w", encoding="utf-8") as file:
        for example_number, ((_, row), prompt, generation) in enumerate(
            zip(df.iterrows(), prompts, generations, strict=True),
            start=1,
        ):
            counters["num_examples"] += 1

            gt_boxed = _normalize_optional_string(row["gt_boxed"])
            pred_boxed = extract_last_boxed(generation)

            if gt_boxed is not None:
                counters["num_gt_boxed"] += 1
            if pred_boxed is not None:
                counters["num_pred_boxed"] += 1

            correct, gold_parseable, prediction_parseable = verify_boxed_answers(
                gt_boxed,
                pred_boxed,
            )

            if not gold_parseable:
                raise RuntimeError(
                    "Prepared evaluation cohort contains a gold answer that is no "
                    "longer parseable. Re-run `posttrain-math data prepare` and "
                    "verify the locked math-verify version."
                )

            counters["num_gt_parseable"] += 1
            if prediction_parseable:
                counters["num_pred_parseable"] += 1
            if correct:
                counters["num_correct"] += 1

            level = str(row["level"])
            problem_type = str(row["type"])
            _update_group(by_level, level, correct=correct)
            _update_group(by_type, problem_type, correct=correct)

            record = {
                "problem": str(row["problem"]),
                "type": problem_type,
                "level": level,
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

            status = "CORRECT" if correct else "INCORRECT"
            gt_display = gt_boxed if gt_boxed is not None else "<none>"
            pred_display = pred_boxed if pred_boxed is not None else "<none>"
            print(
                f"[{example_number}/{num_examples}] {status} | "
                f"gt={gt_display} | pred={pred_display}",
                flush=True,
            )

    actual_num_examples = counters["num_examples"]
    if actual_num_examples != num_examples:
        raise RuntimeError(
            f"Generation count mismatch: expected {num_examples}, got {actual_num_examples}"
        )

    accuracy = (
        counters["num_correct"] / actual_num_examples if actual_num_examples else 0.0
    )
    boxed_output_rate = (
        counters["num_pred_boxed"] / actual_num_examples
        if actual_num_examples
        else 0.0
    )
    parseable_output_rate = (
        counters["num_pred_parseable"] / actual_num_examples
        if actual_num_examples
        else 0.0
    )
    gt_boxed_coverage = (
        counters["num_gt_boxed"] / actual_num_examples
        if actual_num_examples
        else 0.0
    )

    accuracy_by_level = _finalize_groups(by_level)
    accuracy_by_type = _finalize_groups(by_type)

    metrics = {
        "split": split,
        "prompt_strategy": prompt_name,
        "generator": generator.metadata(),
        "generation_config": {
            "do_sample": False,
            "max_new_tokens": max_new_tokens,
        },
        "cohort": {
            "source_rows": source_rows,
            "eligible_rows": eligible_rows,
            "excluded_rows": excluded_rows,
            "evaluated_rows": actual_num_examples,
            "limit": limit,
        },
        "token_diagnostics": token_diagnostics,
        **counters,
        "gt_boxed_coverage": gt_boxed_coverage,
        "boxed_output_rate": boxed_output_rate,
        "parseable_output_rate": parseable_output_rate,
        "accuracy": accuracy,
        "accuracy_by_level": accuracy_by_level,
        "accuracy_by_type": accuracy_by_type,
    }

    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print()
    print("Evaluation")
    print(f"  split:                 {split}")
    print(f"  prompt strategy:       {prompt_name}")
    print(f"  eligible cohort:       {eligible_rows} / {source_rows}")
    print(f"  evaluated examples:    {actual_num_examples}")
    print(f"  correct:               {counters['num_correct']} / {actual_num_examples}")
    print(f"  accuracy:              {accuracy:.2%}")
    print(f"  boxed output rate:     {boxed_output_rate:.2%}")
    print(f"  parseable output rate: {parseable_output_rate:.2%}")

    _print_group_accuracy("Accuracy by level", accuracy_by_level)
    _print_group_accuracy("Accuracy by type", accuracy_by_type)

    print()
    print(f"  predictions: {predictions_path}")
    print(f"  metrics:     {metrics_path}")

    return metrics
