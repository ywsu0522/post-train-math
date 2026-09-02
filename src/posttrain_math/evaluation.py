from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from posttrain_math.answers import (
    extract_last_boxed,
    verify_boxed_answers,
)
from posttrain_math.modeling import (
    TextGenerator,
)
from posttrain_math.prompting import (
    PromptFormatter,
)


def _normalize_optional_string(
    value: Any,
) -> str | None:
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (
        TypeError,
        ValueError,
    ):
        pass

    value = str(value).strip()

    return value or None


def load_eval_split(
    data_dir: Path,
    split: str,
) -> pd.DataFrame:
    if split not in {
        "dev",
        "test",
    }:
        raise ValueError(
            "Evaluation split must "
            "be 'dev' or 'test'"
        )

    path = (
        data_dir
        / f"{split}.parquet"
    )

    if not path.is_file():
        raise FileNotFoundError(
            "Processed split "
            f"not found: {path}"
        )

    df = pd.read_parquet(path)

    required = {
        "problem",
        "solution",
        "type",
        "level",
        "gt_boxed",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Evaluation data missing "
            f"columns: {sorted(missing)}"
        )

    return df


def _length_statistics(
    lengths: list[int],
) -> dict[str, float | int]:
    values = np.asarray(
        lengths,
        dtype=np.int64,
    )

    return {
        "p50": float(
            np.quantile(
                values,
                0.50,
            )
        ),
        "max":
            int(values.max()),
    }


def _print_token_diagnostics(
    *,
    prompt_lengths: list[int],
    solution_lengths: list[int],
    max_new_tokens: int,
) -> dict[str, Any]:
    prompt_stats = (
        _length_statistics(
            prompt_lengths
        )
    )

    solution_stats = (
        _length_statistics(
            solution_lengths
        )
    )

    over_budget = sum(
        length > max_new_tokens
        for length
        in solution_lengths
    )

    ratio = (
        over_budget
        / len(solution_lengths)
        if solution_lengths
        else 0.0
    )

    print("Token diagnostics")
    print()

    print("  Prompt tokens")
    print(
        "    p50: "
        f"{prompt_stats['p50']:.0f}"
    )
    print(
        "    max: "
        f"{prompt_stats['max']}"
    )

    print()
    print(
        "  Gold solution tokens"
    )
    print(
        "    p50: "
        f"{solution_stats['p50']:.0f}"
    )
    print(
        "    max: "
        f"{solution_stats['max']}"
    )

    print()
    print("  Generation budget")
    print(
        "    max_new_tokens: "
        f"{max_new_tokens}"
    )

    print(
        "    gold solution > "
        "max_new_tokens: "
        f"{over_budget} "
        f"({100.0 * ratio:.2f}%)"
    )

    return {
        "prompt_tokens":
            prompt_stats,

        "gold_solution_tokens":
            solution_stats,

        "max_new_tokens":
            max_new_tokens,

        "gold_solution_over_budget":
            {
                "count":
                    over_budget,
                "ratio":
                    ratio,
            },
    }


def evaluate(
    generator: TextGenerator,
    *,
    prompt_formatter:
        PromptFormatter,
    prompt_name: str,
    data_dir: Path,
    split: str,
    output_dir: Path,
    max_new_tokens: int,
    limit: int | None = None,
) -> dict[str, Any]:
    if max_new_tokens <= 0:
        raise ValueError(
            "max_new_tokens must "
            "be positive"
        )

    df = load_eval_split(
        data_dir,
        split,
    )

    if limit is not None:
        if limit <= 0:
            raise ValueError(
                "limit must be positive"
            )

        df = (
            df.head(limit)
            .copy()
        )

    prompts = [
        prompt_formatter(
            str(problem)
        )
        for problem
        in df["problem"]
    ]

    solutions = [
        str(solution)
        for solution
        in df["solution"]
    ]

    prompt_lengths = (
        generator.count_tokens(
            prompts
        )
    )

    solution_lengths = (
        generator.count_tokens(
            solutions
        )
    )

    token_diagnostics = (
        _print_token_diagnostics(
            prompt_lengths=(
                prompt_lengths
            ),
            solution_lengths=(
                solution_lengths
            ),
            max_new_tokens=(
                max_new_tokens
            ),
        )
    )

    num_examples = len(df)

    print()
    print(
        f"Generating "
        f"{num_examples} examples..."
    )
    print()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions_path = (
        output_dir
        / "predictions.jsonl"
    )

    metrics_path = (
        output_dir
        / "metrics.json"
    )

    counters = {
        "num_examples": 0,
        "num_gt_boxed": 0,
        "num_gt_parseable": 0,
        "num_pred_boxed": 0,
        "num_pred_parseable": 0,
        "num_correct": 0,
    }

    generations = (
        generator.iter_generate(
            prompts,
            max_new_tokens=(
                max_new_tokens
            ),
            do_sample=False,
        )
    )

    with predictions_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for (
            example_number,
            (
                (_, row),
                prompt,
                generation,
            ),
        ) in enumerate(
            zip(
                df.iterrows(),
                prompts,
                generations,
                strict=True,
            ),
            start=1,
        ):
            counters[
                "num_examples"
            ] += 1

            gt_boxed = (
                _normalize_optional_string(
                    row["gt_boxed"]
                )
            )

            pred_boxed = (
                extract_last_boxed(
                    generation
                )
            )

            if gt_boxed is not None:
                counters[
                    "num_gt_boxed"
                ] += 1

            if pred_boxed is not None:
                counters[
                    "num_pred_boxed"
                ] += 1

            (
                correct,
                gold_parseable,
                prediction_parseable,
            ) = verify_boxed_answers(
                gt_boxed,
                pred_boxed,
            )

            if gold_parseable:
                counters[
                    "num_gt_parseable"
                ] += 1

            if prediction_parseable:
                counters[
                    "num_pred_parseable"
                ] += 1

            if correct:
                counters[
                    "num_correct"
                ] += 1

            record = {
                "problem":
                    str(
                        row["problem"]
                    ),

                "type":
                    str(
                        row["type"]
                    ),

                "level":
                    str(
                        row["level"]
                    ),

                "prompt_strategy":
                    prompt_name,

                "prompt":
                    prompt,

                "gt_boxed":
                    gt_boxed,

                "generation":
                    generation,

                "pred_boxed":
                    pred_boxed,

                "gold_parseable":
                    gold_parseable,

                "prediction_parseable":
                    prediction_parseable,

                "correct":
                    correct,
            }

            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

            # Preserve partial results even
            # if a later example fails.
            file.flush()

            status = (
                "CORRECT"
                if correct
                else "INCORRECT"
            )

            gt_display = (
                gt_boxed
                if gt_boxed is not None
                else "<none>"
            )

            pred_display = (
                pred_boxed
                if pred_boxed is not None
                else "<none>"
            )

            print(
                f"[{example_number}/"
                f"{num_examples}] "
                f"{status} | "
                f"gt={gt_display} | "
                f"pred={pred_display}",
                flush=True,
            )

    actual_num_examples = (
        counters["num_examples"]
    )

    num_gt_parseable = (
        counters[
            "num_gt_parseable"
        ]
    )

    accuracy = (
        counters["num_correct"]
        / num_gt_parseable
        if num_gt_parseable
        else 0.0
    )

    gt_boxed_coverage = (
        counters["num_gt_boxed"]
        / actual_num_examples
        if actual_num_examples
        else 0.0
    )

    boxed_output_rate = (
        counters["num_pred_boxed"]
        / actual_num_examples
        if actual_num_examples
        else 0.0
    )

    metrics = {
        "split":
            split,

        "prompt_strategy":
            prompt_name,

        "generator":
            generator.metadata(),

        "generation_config": {
            "do_sample":
                False,
            "max_new_tokens":
                max_new_tokens,
        },

        "token_diagnostics":
            token_diagnostics,

        **counters,

        "gt_boxed_coverage":
            gt_boxed_coverage,

        "boxed_output_rate":
            boxed_output_rate,

        "accuracy":
            accuracy,
    }

    metrics_path.write_text(
        json.dumps(
            metrics,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("Evaluation")
    print(
        f"  split:             "
        f"{split}"
    )
    print(
        "  prompt strategy:   "
        f"{prompt_name}"
    )
    print(
        "  examples:          "
        f"{actual_num_examples}"
    )
    print(
        "  GT boxed coverage: "
        f"{gt_boxed_coverage:.2%}"
    )
    print(
        "  model boxed rate:  "
        f"{boxed_output_rate:.2%}"
    )
    print(
        "  correct:           "
        f"{counters['num_correct']}"
    )
    print(
        "  accuracy:          "
        f"{accuracy:.2%}"
    )

    print()
    print(
        "  predictions: "
        f"{predictions_path}"
    )
    print(
        "  metrics:     "
        f"{metrics_path}"
    )

    return metrics
