from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import datasets as hf_datasets
import pandas as pd
from huggingface_hub import HfApi
from sklearn.model_selection import (
    train_test_split,
)
from tabulate import tabulate

from postrain_math_lab.answers import (
    classify_boxed_format,
    extract_last_boxed,
)


ORIGINAL_COLUMNS = [
    "problem",
    "solution",
    "type",
    "level",
]

EXPECTED_RAW_TRAIN_ROWS = 7500
EXPECTED_RAW_TEST_ROWS = 5000

HENDRYCKS_MATH_REPO = "EleutherAI/hendrycks_math"
HENDRYCKS_MATH_CONFIGS = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)


def download_raw_datasets(
    *,
    output_dir: Path,
    repo_id: str = HENDRYCKS_MATH_REPO,
    revision: str = "main",
    force: bool = False,
) -> tuple[Path, Path]:
    """Materialize the Hugging Face MATH dataset into two local parquet files."""
    output_dir = Path(output_dir)
    train_path = output_dir / "math_train.parquet"
    test_path = output_dir / "math_test.parquet"

    existing = [path for path in (train_path, test_path) if path.exists()]
    if existing and not force:
        manifest_path = output_dir / "download_manifest.json"
        if train_path.is_file() and test_path.is_file() and manifest_path.is_file():
            print("Raw MATH dataset already present; skipping download")
            print(f"- train: {train_path}")
            print(f"- test: {test_path}")
            return train_path, test_path
        raise FileExistsError(
            "Raw dataset is partially present: "
            + ", ".join(str(path) for path in existing)
            + ". Use --force to replace it."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    info = HfApi().dataset_info(repo_id=repo_id, revision=revision)
    commit_sha = info.sha
    if not commit_sha:
        raise RuntimeError(
            f"Could not resolve a commit SHA for {repo_id}@{revision}"
        )

    split_frames: dict[str, list[pd.DataFrame]] = {"train": [], "test": []}

    for config_name in HENDRYCKS_MATH_CONFIGS:
        dataset = hf_datasets.load_dataset(
            repo_id,
            config_name,
            revision=commit_sha,
        )
        for split in ("train", "test"):
            frame = dataset[split].to_pandas()
            validate_required_columns(frame, f"{config_name}/{split}")
            split_frames[split].append(frame[ORIGINAL_COLUMNS].copy())

    train_df = pd.concat(split_frames["train"], ignore_index=True)
    test_df = pd.concat(split_frames["test"], ignore_index=True)

    if len(train_df) != EXPECTED_RAW_TRAIN_ROWS:
        raise RuntimeError(
            f"Unexpected raw train rows: {len(train_df)} "
            f"(expected {EXPECTED_RAW_TRAIN_ROWS})"
        )
    if len(test_df) != EXPECTED_RAW_TEST_ROWS:
        raise RuntimeError(
            f"Unexpected raw test rows: {len(test_df)} "
            f"(expected {EXPECTED_RAW_TEST_ROWS})"
        )

    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)

    manifest = {
        "repo_id": repo_id,
        "requested_revision": revision,
        "resolved_commit": commit_sha,
        "configs": list(HENDRYCKS_MATH_CONFIGS),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_path": str(train_path),
        "test_path": str(test_path),
    }
    (output_dir / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("Raw MATH dataset downloaded")
    print(f"- repo: {repo_id}")
    print(f"- revision: {revision}")
    print(f"- commit: {commit_sha}")
    print(f"- train: {train_path} ({len(train_df)} rows)")
    print(f"- test: {test_path} ({len(test_df)} rows)")

    return train_path, test_path


def load_dataset(
    path: Path,
) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    return pd.read_parquet(path)


def validate_required_columns(
    df: pd.DataFrame,
    name: str,
) -> None:
    missing = (
        set(ORIGINAL_COLUMNS)
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"{name}: missing required "
            f"columns: {sorted(missing)}"
        )


def _column_error_counts(
    series: pd.Series,
) -> dict[str, int]:
    na_mask = series.isna()

    non_null = (
        series[~na_mask]
        .astype(str)
    )

    empty_mask = (
        non_null.eq("")
    )

    whitespace_mask = (
        non_null.str.strip().eq("")
        & ~empty_mask
    )

    return {
        "na": int(na_mask.sum()),
        "empty": int(
            empty_mask.sum()
        ),
        "whitespace_only": int(
            whitespace_mask.sum()
        ),
    }


def _print_raw_dataset_report(
    name: str,
    df: pd.DataFrame,
) -> None:
    print(name)
    print(f"  rows: {len(df)}")

    duplicate_rows = int(
        df.duplicated(
            subset=ORIGINAL_COLUMNS,
            keep="first",
        ).sum()
    )

    print(
        "  duplicated rows: "
        f"{duplicate_rows}"
    )

    print("  error values:")

    for column in ORIGINAL_COLUMNS:
        counts = (
            _column_error_counts(
                df[column]
            )
        )

        print(f"    {column}:")
        print(
            "      NA:              "
            f"{counts['na']}"
        )
        print(
            '      empty "":        '
            f"{counts['empty']}"
        )
        print(
            "      whitespace-only: "
            f"{counts['whitespace_only']}"
        )


def inspect_raw_datasets(
    train_path: Path,
    test_path: Path,
) -> None:
    raw_train = load_dataset(
        train_path
    )

    raw_test = load_dataset(
        test_path
    )

    validate_required_columns(
        raw_train,
        "Raw train",
    )

    validate_required_columns(
        raw_test,
        "Raw test",
    )

    _print_raw_dataset_report(
        "Raw train",
        raw_train,
    )

    print()

    _print_raw_dataset_report(
        "Raw test",
        raw_test,
    )

    train_problems = set(
        raw_train["problem"]
        .dropna()
        .astype(str)
    )

    test_problems = set(
        raw_test["problem"]
        .dropna()
        .astype(str)
    )

    overlap = (
        train_problems
        & test_problems
    )

    print()
    print("Raw train/test")
    print(
        "  exact problem overlap: "
        f"{len(overlap)} "
        f"[{'PASS' if not overlap else 'FAIL'}]"
    )


def _assert_preparable(
    df: pd.DataFrame,
    name: str,
) -> None:
    validate_required_columns(
        df,
        name,
    )

    for column in ORIGINAL_COLUMNS:
        counts = (
            _column_error_counts(
                df[column]
            )
        )

        if any(counts.values()):
            raise ValueError(
                f"{name}: invalid values "
                f"in column '{column}': "
                f"{counts}"
            )


def add_gt_boxed(
    df: pd.DataFrame,
) -> pd.DataFrame:
    result = df.copy()

    result["gt_boxed"] = (
        result["solution"]
        .astype(str)
        .map(extract_last_boxed)
    )

    return result


def split_raw_train(
    df: pd.DataFrame,
    *,
    seed: int,
    dev_ratio: float,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    if not 0.0 < dev_ratio < 1.0:
        raise ValueError(
            "dev_ratio must be "
            "between 0 and 1"
        )

    strata = (
        df["type"].astype(str)
        + "||"
        + df["level"].astype(str)
    )

    train_df, dev_df = (
        train_test_split(
            df,
            test_size=dev_ratio,
            random_state=seed,
            shuffle=True,
            stratify=strata,
        )
    )

    return (
        train_df.reset_index(
            drop=True
        ),
        dev_df.reset_index(
            drop=True
        ),
    )


def _same_rows_unordered(
    left: pd.DataFrame,
    right: pd.DataFrame,
    columns: list[str],
) -> bool:
    left_hashes = (
        pd.util.hash_pandas_object(
            left[columns],
            index=False,
        )
        .value_counts()
        .sort_index()
    )

    right_hashes = (
        pd.util.hash_pandas_object(
            right[columns],
            index=False,
        )
        .value_counts()
        .sort_index()
    )

    return left_hashes.equals(
        right_hashes
    )


def _joint_distribution(
    df: pd.DataFrame,
) -> dict[
    tuple[str, str],
    float,
]:
    counts = (
        df.groupby(
            [
                "level",
                "type",
            ],
            dropna=False,
        )
        .size()
    )

    total = len(df)

    return {
        (
            str(level),
            str(problem_type),
        ):
        100.0 * int(count) / total
        for (
            level,
            problem_type,
        ), count
        in counts.items()
    }


def _max_distribution_gap(
    reference: dict[
        tuple[str, str],
        float,
    ],
    target: dict[
        tuple[str, str],
        float,
    ],
) -> tuple[
    tuple[str, str],
    float,
    float,
    float,
]:
    keys = sorted(
        set(reference)
        | set(target)
    )

    if not keys:
        raise ValueError(
            "Distribution is empty"
        )

    best_key = keys[0]
    best_reference = reference.get(
        best_key,
        0.0,
    )
    best_target = target.get(
        best_key,
        0.0,
    )
    best_gap = abs(
        best_reference
        - best_target
    )

    for key in keys[1:]:
        reference_value = (
            reference.get(key, 0.0)
        )

        target_value = (
            target.get(key, 0.0)
        )

        gap = abs(
            reference_value
            - target_value
        )

        if gap > best_gap:
            best_key = key
            best_reference = (
                reference_value
            )
            best_target = (
                target_value
            )
            best_gap = gap

    return (
        best_key,
        best_reference,
        best_target,
        best_gap,
    )


def print_type_level_distribution(
    raw: pd.DataFrame,
    train: pd.DataFrame,
    dev: pd.DataFrame,
) -> None:
    raw_dist = (
        _joint_distribution(raw)
    )

    train_dist = (
        _joint_distribution(train)
    )

    dev_dist = (
        _joint_distribution(dev)
    )

    preferred_levels = [
        f"Level {index}"
        for index in range(1, 6)
    ]

    observed_levels = {
        str(value)
        for value
        in raw["level"].unique()
    }

    levels = [
        level
        for level in preferred_levels
        if level in observed_levels
    ]

    levels.extend(
        sorted(
            observed_levels
            - set(preferred_levels)
        )
    )

    problem_types = sorted(
        {
            str(value)
            for value
            in raw["type"].unique()
        }
    )

    headers = [
        "Type",
        *levels,
    ]

    table_rows: list[
        list[str]
    ] = []

    for problem_type in problem_types:
        row = [problem_type]

        for level in levels:
            percentage = (
                raw_dist.get(
                    (
                        level,
                        problem_type,
                    ),
                    0.0,
                )
            )

            row.append(
                f"{percentage:.2f}%"
            )

        table_rows.append(row)

    print(
        "Type × Level distribution "
        "— raw train"
    )
    print()

    print(
        tabulate(
            table_rows,
            headers=headers,
            tablefmt="rounded_grid",
            stralign="center",
            disable_numparse=True,
        )
    )

    (
        train_key,
        raw_train_pct,
        train_pct,
        train_gap,
    ) = _max_distribution_gap(
        raw_dist,
        train_dist,
    )

    (
        dev_key,
        raw_dev_pct,
        dev_pct,
        dev_gap,
    ) = _max_distribution_gap(
        raw_dist,
        dev_dist,
    )

    train_level, train_type = (
        train_key
    )

    dev_level, dev_type = (
        dev_key
    )

    print()
    print(
        "- Largest raw/train gap: "
        f"{train_type} × {train_level} "
        f"= {train_gap:.2f} percentage points "
        f"(raw {raw_train_pct:.2f}%, "
        f"train {train_pct:.2f}%)"
    )

    print(
        "- Largest raw/dev gap: "
        f"{dev_type} × {dev_level} "
        f"= {dev_gap:.2f} percentage points "
        f"(raw {raw_dev_pct:.2f}%, "
        f"dev {dev_pct:.2f}%)"
    )


def _boxed_audit_counts(
    df: pd.DataFrame,
) -> tuple[
    dict[str, int],
    dict[str, list[int]],
]:
    categories = {
        "single_valid": 0,
        "multiple_valid": 0,
        "empty_boxed": 0,
        "malformed_unbraced": 0,
        "other_malformed": 0,
        "no_boxed": 0,
    }

    indices = {
        category: []
        for category
        in categories
    }

    for index, solution in enumerate(
        df["solution"].astype(str)
    ):
        category = (
            classify_boxed_format(
                solution
            )
        )

        categories[category] += 1

        indices[category].append(
            index
        )

    return (
        categories,
        indices,
    )


def _format_count(
    count: int,
    total: int,
) -> str:
    ratio = (
        100.0 * count / total
        if total
        else 0.0
    )

    return (
        f"{count} ({ratio:.2f}%)"
    )


def _one_line(
    text: str,
) -> str:
    """
    Collapse embedded whitespace/newlines
    for stable one-line CLI output.
    """
    return " ".join(
        text.split()
    )


def print_raw_boxed_audit(
    name: str,
    df: pd.DataFrame,
    *,
    seed: int,
) -> None:
    counts, indices = (
        _boxed_audit_counts(df)
    )

    labels = {
        "single_valid":
            r"single valid \boxed{...}",
        "multiple_valid":
            r"multiple valid \boxed{...}",
        "empty_boxed":
            r"empty \boxed{}",
        "malformed_unbraced":
            r"malformed \boxed ANSWER",
        "other_malformed":
            "other malformed boxed",
        "no_boxed":
            "no boxed",
    }

    total = len(df)

    rng = random.Random(seed)

    sample_categories = {
        "empty_boxed",
        "malformed_unbraced",
        "other_malformed",
    }

    print(
        f"GT boxed format audit — {name}"
    )

    for (
        category,
        label,
    ) in labels.items():
        count = counts[category]

        line = (
            f"- {label}: "
            f"{_format_count(count, total)}"
        )

        if (
            count > 0
            and category
            in sample_categories
        ):
            sample_index = (
                rng.choice(
                    indices[category]
                )
            )

            solution = _one_line(
                str(
                    df.iloc[
                        sample_index
                    ]["solution"]
                )
            )

            line += (
                f' | [{sample_index}] '
                f'"{solution}"'
            )

        print(line)


def print_processed_boxed_report(
    name: str,
    df: pd.DataFrame,
) -> None:
    valid = int(
        df["gt_boxed"]
        .notna()
        .sum()
    )

    missing = (
        len(df) - valid
    )

    print(
        f"GT boxed extraction — {name}"
    )

    print(
        "- valid gt_boxed: "
        f"{_format_count(valid, len(df))}"
    )

    print(
        "- missing gt_boxed: "
        f"{_format_count(missing, len(df))}"
    )


def _processed_invariants(
    raw_train: pd.DataFrame,
    raw_test: pd.DataFrame,
    train: pd.DataFrame,
    dev: pd.DataFrame,
    test: pd.DataFrame,
) -> dict[str, bool]:
    train_problems = set(
        train["problem"].astype(str)
    )

    dev_problems = set(
        dev["problem"].astype(str)
    )

    recovered = pd.concat(
        [
            train,
            dev,
        ],
        ignore_index=True,
    )

    raw_train_columns = list(
        raw_train.columns
    )

    raw_test_columns = list(
        raw_test.columns
    )

    test_schema_ok = (
        list(test.columns)
        == [
            *raw_test_columns,
            "gt_boxed",
        ]
    )

    test_values_ok = (
        test[raw_test_columns]
        .reset_index(drop=True)
        .equals(
            raw_test[
                raw_test_columns
            ]
            .reset_index(drop=True)
        )
    )

    return {
        "train_dev_disjoint":
            not (
                train_problems
                & dev_problems
            ),

        "recover_raw_train":
            _same_rows_unordered(
                recovered,
                raw_train,
                columns=(
                    raw_train_columns
                ),
            ),

        "gt_boxed_exists":
            all(
                "gt_boxed"
                in frame.columns
                for frame
                in (
                    train,
                    dev,
                    test,
                )
            ),

        "raw_test_preserved":
            (
                test_schema_ok
                and test_values_ok
            ),
    }


def prepare_datasets(
    train_path: Path,
    test_path: Path,
    output_dir: Path,
    *,
    seed: int,
    dev_ratio: float,
) -> None:
    raw_train = load_dataset(
        train_path
    )

    raw_test = load_dataset(
        test_path
    )

    _assert_preparable(
        raw_train,
        "Raw train",
    )

    _assert_preparable(
        raw_test,
        "Raw test",
    )

    if (
        len(raw_train)
        != EXPECTED_RAW_TRAIN_ROWS
    ):
        raise ValueError(
            "Raw train row count is "
            f"{len(raw_train)}, expected "
            f"{EXPECTED_RAW_TRAIN_ROWS}"
        )

    if (
        len(raw_test)
        != EXPECTED_RAW_TEST_ROWS
    ):
        raise ValueError(
            "Raw test row count is "
            f"{len(raw_test)}, expected "
            f"{EXPECTED_RAW_TEST_ROWS}"
        )

    overlap = (
        set(
            raw_train[
                "problem"
            ].astype(str)
        )
        & set(
            raw_test[
                "problem"
            ].astype(str)
        )
    )

    if overlap:
        raise ValueError(
            "Raw train/test problem "
            f"overlap detected: "
            f"{len(overlap)}"
        )

    print_raw_boxed_audit(
        "raw train",
        raw_train,
        seed=seed,
    )

    print()

    print_raw_boxed_audit(
        "raw test",
        raw_test,
        seed=seed,
    )

    processed_train_pool = (
        add_gt_boxed(raw_train)
    )

    processed_test = (
        add_gt_boxed(raw_test)
    )

    train_df, dev_df = (
        split_raw_train(
            processed_train_pool,
            seed=seed,
            dev_ratio=dev_ratio,
        )
    )

    test_df = (
        processed_test
        .reset_index(drop=True)
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_path_out = (
        output_dir
        / "train.parquet"
    )

    dev_path_out = (
        output_dir
        / "dev.parquet"
    )

    test_path_out = (
        output_dir
        / "test.parquet"
    )

    manifest_path = (
        output_dir
        / "manifest.json"
    )

    train_df.to_parquet(
        train_path_out,
        index=False,
    )

    dev_df.to_parquet(
        dev_path_out,
        index=False,
    )

    test_df.to_parquet(
        test_path_out,
        index=False,
    )

    invariants = (
        _processed_invariants(
            raw_train,
            raw_test,
            train_df,
            dev_df,
            test_df,
        )
    )

    if not all(
        invariants.values()
    ):
        failed = [
            name
            for name, passed
            in invariants.items()
            if not passed
        ]

        raise RuntimeError(
            "Processed invariant failure: "
            f"{failed}"
        )

    print()
    print("Rows")
    print(
        f"- raw train: {len(raw_train)}"
    )
    print(
        f"- train:     {len(train_df)}"
    )
    print(
        f"- dev:       {len(dev_df)}"
    )
    print(
        f"- raw test:  {len(raw_test)}"
    )
    print(
        f"- test:      {len(test_df)}"
    )

    print()
    print("Processed invariants")
    print(
        "- train/dev disjoint: [PASS]"
    )
    print(
        "- train + dev recover "
        "raw train: [PASS]"
    )
    print(
        "- gt_boxed exists in "
        "train/dev/test: [PASS]"
    )
    print(
        "- raw test == processed test "
        "except gt_boxed: [PASS]"
    )

    print()

    print_type_level_distribution(
        processed_train_pool,
        train_df,
        dev_df,
    )

    print()

    print_processed_boxed_report(
        "train",
        train_df,
    )

    print()

    print_processed_boxed_report(
        "dev",
        dev_df,
    )

    print()

    print_processed_boxed_report(
        "test",
        test_df,
    )

    manifest: dict[str, Any] = {
        "raw": {
            "train_rows":
                len(raw_train),
            "test_rows":
                len(raw_test),
        },
        "split": {
            "seed": seed,
            "dev_ratio":
                dev_ratio,
            "stratify": [
                "type",
                "level",
            ],
        },
        "processed": {
            "train_rows":
                len(train_df),
            "dev_rows":
                len(dev_df),
            "test_rows":
                len(test_df),
        },
        "invariants":
            invariants,
    }

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("Artifacts")
    print(f"- {train_path_out}")
    print(f"- {dev_path_out}")
    print(f"- {test_path_out}")
    print(f"- {manifest_path}")