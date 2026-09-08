from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
from sklearn.model_selection import train_test_split

from posttrain_math.answers import RationalGold, classify_exact_rational_solution

ORIGINAL_COLUMNS = ["problem", "solution", "type", "level"]
COHORT_COLUMNS = [
    "gt_boxed",
    "gt_numerator",
    "gt_denominator",
    "rational_eligible",
    "rational_exclusion",
]

EXPECTED_RAW_TRAIN_ROWS = 7500
EXPECTED_RAW_TEST_ROWS = 5000
EXPECTED_EXACT_RATIONAL_TRAIN_ROWS = 5448
EXPECTED_EXACT_RATIONAL_TEST_ROWS = 3592

MATH_DATASET_REPO = "DigitalLearningGmbH/MATH-lighteval"
MATH_DATASET_REVISION = "f06834690385b29df31ccc717250746a3ba0322b"
MATH_TRAIN_FILE = "data/train-00000-of-00001.parquet"
MATH_TEST_FILE = "data/test-00000-of-00001.parquet"
MATH_TRAIN_SHA256 = "eca6e667f4305dd5e5ba09b4fd55e7f3174a0fbe361cdfd4c44758b593a76933"
MATH_TEST_SHA256 = "7dca8d6e41af88ecf82f2b5f36eb5530e083aaaa86ee325f62bd5c31535178c6"
COHORT_NAME = "exact-rational-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_download_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Could not read raw dataset manifest: {path}"
        ) from exc

    if not isinstance(value, dict):
        raise TypeError(
            f"Raw dataset manifest must contain a JSON object: {path}"
        )
    return value


def _validate_existing_raw_dataset(
    *,
    train_path: Path,
    test_path: Path,
    manifest_path: Path,
    repo_id: str,
    revision: str,
) -> None:
    manifest = _read_download_manifest(manifest_path)
    errors: list[str] = []

    if manifest.get("repo_id") != repo_id:
        errors.append(
            "repo_id mismatch: "
            f"{manifest.get('repo_id')!r} != {repo_id!r}"
        )

    if manifest.get("requested_revision") != revision:
        errors.append(
            "requested_revision mismatch: "
            f"{manifest.get('requested_revision')!r} != {revision!r}"
        )

    expected_source_files = {
        "train": MATH_TRAIN_FILE,
        "test": MATH_TEST_FILE,
    }
    if manifest.get("source_files") != expected_source_files:
        errors.append(
            "source_files mismatch: "
            f"{manifest.get('source_files')!r} != {expected_source_files!r}"
        )

    sha256_manifest = manifest.get("sha256")
    if not isinstance(sha256_manifest, dict):
        errors.append("manifest is missing sha256 metadata")
        sha256_manifest = {}

    actual_train_sha256 = _sha256(train_path)
    actual_test_sha256 = _sha256(test_path)

    manifest_train_sha256 = sha256_manifest.get("train")
    manifest_test_sha256 = sha256_manifest.get("test")

    if manifest_train_sha256 != actual_train_sha256:
        errors.append(
            "train SHA256 mismatch: "
            f"{actual_train_sha256} != {manifest_train_sha256}"
        )

    if manifest_test_sha256 != actual_test_sha256:
        errors.append(
            "test SHA256 mismatch: "
            f"{actual_test_sha256} != {manifest_test_sha256}"
        )

    if repo_id == MATH_DATASET_REPO and revision == MATH_DATASET_REVISION:
        if actual_train_sha256 != MATH_TRAIN_SHA256:
            errors.append(
                "pinned train SHA256 mismatch: "
                f"{actual_train_sha256} != {MATH_TRAIN_SHA256}"
            )

        if actual_test_sha256 != MATH_TEST_SHA256:
            errors.append(
                "pinned test SHA256 mismatch: "
                f"{actual_test_sha256} != {MATH_TEST_SHA256}"
            )

    if errors:
        details = "\n".join(f"- {error}" for error in errors)
        raise RuntimeError(
            "Existing raw MATH dataset does not match the requested source:\n"
            f"{details}\n"
            "Use --force to replace the local raw dataset."
        )


def download_raw_datasets(
    *,
    output_dir: Path,
    repo_id: str = MATH_DATASET_REPO,
    revision: str = MATH_DATASET_REVISION,
    force: bool = False,
) -> tuple[Path, Path]:
    """Materialize the consolidated MATH train/test Parquet files locally."""
    output_dir = Path(output_dir)
    train_path = output_dir / "math_train.parquet"
    test_path = output_dir / "math_test.parquet"
    manifest_path = output_dir / "download_manifest.json"

    existing = [path for path in (train_path, test_path, manifest_path) if path.exists()]
    complete = train_path.is_file() and test_path.is_file() and manifest_path.is_file()
    if complete and not force:
        _validate_existing_raw_dataset(
            train_path=train_path,
            test_path=test_path,
            manifest_path=manifest_path,
            repo_id=repo_id,
            revision=revision,
        )
        print("Raw MATH dataset already present and verified; skipping download")
        print(f"- train: {train_path}")
        print(f"- test: {test_path}")
        return train_path, test_path
    if existing and not force:
        raise FileExistsError(
            "Raw dataset is partially present: "
            + ", ".join(str(path) for path in existing)
            + ". Use --force to replace it."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    info = HfApi().dataset_info(repo_id=repo_id, revision=revision)
    commit_sha = info.sha
    if not commit_sha:
        raise RuntimeError(f"Could not resolve a commit SHA for {repo_id}@{revision}")

    cached_train = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=MATH_TRAIN_FILE,
            repo_type="dataset",
            revision=commit_sha,
        )
    )
    cached_test = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=MATH_TEST_FILE,
            repo_type="dataset",
            revision=commit_sha,
        )
    )

    shutil.copyfile(cached_train, train_path)
    shutil.copyfile(cached_test, test_path)

    train_sha256 = _sha256(train_path)
    test_sha256 = _sha256(test_path)
    if repo_id == MATH_DATASET_REPO and revision == MATH_DATASET_REVISION:
        if train_sha256 != MATH_TRAIN_SHA256:
            raise RuntimeError(
                "Pinned train Parquet SHA256 mismatch: "
                f"{train_sha256} != {MATH_TRAIN_SHA256}"
            )
        if test_sha256 != MATH_TEST_SHA256:
            raise RuntimeError(
                "Pinned test Parquet SHA256 mismatch: "
                f"{test_sha256} != {MATH_TEST_SHA256}"
            )

    train_df = load_dataset(train_path)
    test_df = load_dataset(test_path)
    validate_required_columns(train_df, "Raw train")
    validate_required_columns(test_df, "Raw test")
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

    manifest = {
        "repo_id": repo_id,
        "requested_revision": revision,
        "resolved_commit": commit_sha,
        "source_files": {
            "train": MATH_TRAIN_FILE,
            "test": MATH_TEST_FILE,
        },
        "sha256": {
            "train": train_sha256,
            "test": test_sha256,
        },
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_path": str(train_path),
        "test_path": str(test_path),
    }
    manifest_path.write_text(
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


def load_dataset(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_parquet(path)


def validate_required_columns(df: pd.DataFrame, name: str) -> None:
    missing = set(ORIGINAL_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{name}: missing required columns: {sorted(missing)}")


def _column_error_counts(series: pd.Series) -> dict[str, int]:
    na_mask = series.isna()
    non_null = series[~na_mask].astype(str)
    empty_mask = non_null.eq("")
    whitespace_mask = non_null.str.strip().eq("") & ~empty_mask
    return {
        "na": int(na_mask.sum()),
        "empty": int(empty_mask.sum()),
        "whitespace_only": int(whitespace_mask.sum()),
    }


def _assert_preparable(df: pd.DataFrame, name: str) -> None:
    validate_required_columns(df, name)
    for column in ORIGINAL_COLUMNS:
        counts = _column_error_counts(df[column])
        if any(counts.values()):
            raise ValueError(f"{name}: invalid values in column '{column}': {counts}")


def inspect_raw_datasets(train_path: Path, test_path: Path) -> None:
    raw_train = load_dataset(train_path)
    raw_test = load_dataset(test_path)
    _assert_preparable(raw_train, "Raw train")
    _assert_preparable(raw_test, "Raw test")

    for name, df in (("Raw train", raw_train), ("Raw test", raw_test)):
        duplicate_rows = int(df.duplicated(subset=ORIGINAL_COLUMNS, keep="first").sum())
        cohort = annotate_rational_cohort(df)
        eligible = int(cohort["rational_eligible"].sum())
        print(name)
        print(f"  rows: {len(df)}")
        print(f"  duplicated rows: {duplicate_rows}")
        print(f"  {COHORT_NAME}: {eligible} ({eligible / len(df):.2%})")
        print(f"  exclusions: {_exclusion_counts(cohort)}")
        print()

    overlap = set(raw_train["problem"].astype(str)) & set(raw_test["problem"].astype(str))
    print("Raw train/test")
    print(f"  exact problem overlap: {len(overlap)} [{'PASS' if not overlap else 'FAIL'}]")


def _classification_columns(classification: RationalGold) -> tuple[Any, ...]:
    return (
        classification.gt_boxed,
        classification.numerator,
        classification.denominator,
        classification.eligible,
        classification.exclusion_reason,
    )


def annotate_rational_cohort(df: pd.DataFrame) -> pd.DataFrame:
    """Annotate rows with deterministic exact-rational-v1 gold metadata."""
    result = df.copy()
    classifications = [
        classify_exact_rational_solution(str(solution))
        for solution in result["solution"]
    ]
    columns = [_classification_columns(item) for item in classifications]

    result["gt_boxed"] = [row[0] for row in columns]
    result["gt_numerator"] = pd.array([row[1] for row in columns], dtype="Int64")
    result["gt_denominator"] = pd.array([row[2] for row in columns], dtype="Int64")
    result["rational_eligible"] = [bool(row[3]) for row in columns]
    result["rational_exclusion"] = [row[4] for row in columns]
    return result


def split_raw_train(
    df: pd.DataFrame,
    *,
    seed: int,
    dev_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 < dev_ratio < 1.0:
        raise ValueError("dev_ratio must be between 0 and 1")
    strata = df["type"].astype(str) + "||" + df["level"].astype(str)
    train_df, dev_df = train_test_split(
        df,
        test_size=dev_ratio,
        random_state=seed,
        shuffle=True,
        stratify=strata,
    )
    return train_df.reset_index(drop=True), dev_df.reset_index(drop=True)


def _same_rows_unordered(
    left: pd.DataFrame,
    right: pd.DataFrame,
    columns: list[str],
) -> bool:
    left_hashes = pd.util.hash_pandas_object(left[columns], index=False).value_counts().sort_index()
    right_hashes = pd.util.hash_pandas_object(right[columns], index=False).value_counts().sort_index()
    return left_hashes.equals(right_hashes)


def _exclusion_counts(df: pd.DataFrame) -> dict[str, int]:
    values = df.loc[~df["rational_eligible"], "rational_exclusion"].dropna().astype(str)
    return dict(sorted(Counter(values).items()))


def _cohort_cell_stats(df: pd.DataFrame) -> dict[str, Any]:
    source_counts = df.groupby(["type", "level"], dropna=False).size()
    cohort_df = df[df["rational_eligible"]]
    cohort_counts = cohort_df.groupby(["type", "level"], dropna=False).size()
    source_total = len(df)
    cohort_total = len(cohort_df)

    cells: list[dict[str, Any]] = []
    abs_drift_sum = 0.0
    max_abs_drift = 0.0
    for problem_type, level in sorted(source_counts.index, key=lambda key: (str(key[0]), str(key[1]))):
        source_count = int(source_counts.loc[(problem_type, level)])
        cohort_count = int(cohort_counts.get((problem_type, level), 0))
        source_share = 100.0 * source_count / source_total if source_total else 0.0
        cohort_share = 100.0 * cohort_count / cohort_total if cohort_total else 0.0
        drift = cohort_share - source_share
        abs_drift_sum += abs(drift)
        max_abs_drift = max(max_abs_drift, abs(drift))
        cells.append(
            {
                "type": str(problem_type),
                "level": str(level),
                "source_count": source_count,
                "cohort_count": cohort_count,
                "retention": cohort_count / source_count if source_count else 0.0,
                "source_share_pct": source_share,
                "cohort_share_pct": cohort_share,
                "share_drift_pp": drift,
            }
        )

    return {
        "source_rows": source_total,
        "cohort_rows": cohort_total,
        "retention": cohort_total / source_total if source_total else 0.0,
        "max_abs_share_drift_pp": max_abs_drift,
        "total_variation_shift_pct": abs_drift_sum / 2.0,
        "cells": cells,
    }


def _cohort_summary(df: pd.DataFrame) -> dict[str, Any]:
    eligible = int(df["rational_eligible"].sum())
    return {
        "source_rows": len(df),
        "eligible_rows": eligible,
        "excluded_rows": len(df) - eligible,
        "retention": eligible / len(df) if len(df) else 0.0,
        "exclusion_counts": _exclusion_counts(df),
        "type_level": _cohort_cell_stats(df),
    }


def prepare_datasets(
    train_path: Path,
    test_path: Path,
    output_dir: Path,
    *,
    seed: int,
    dev_ratio: float,
) -> None:
    raw_train = load_dataset(train_path)
    raw_test = load_dataset(test_path)
    _assert_preparable(raw_train, "Raw train")
    _assert_preparable(raw_test, "Raw test")

    if len(raw_train) != EXPECTED_RAW_TRAIN_ROWS:
        raise ValueError(
            f"Raw train row count is {len(raw_train)}, expected {EXPECTED_RAW_TRAIN_ROWS}"
        )
    if len(raw_test) != EXPECTED_RAW_TEST_ROWS:
        raise ValueError(
            f"Raw test row count is {len(raw_test)}, expected {EXPECTED_RAW_TEST_ROWS}"
        )

    overlap = set(raw_train["problem"].astype(str)) & set(raw_test["problem"].astype(str))
    if overlap:
        raise ValueError(f"Raw train/test problem overlap detected: {len(overlap)}")

    annotated_train = annotate_rational_cohort(raw_train)
    annotated_test = annotate_rational_cohort(raw_test).reset_index(drop=True)

    canonical_source = (
        len(raw_train) == 7500
        and len(raw_test) == 5000
    )
    if canonical_source:
        train_cohort_rows = int(annotated_train["rational_eligible"].sum())
        test_cohort_rows = int(annotated_test["rational_eligible"].sum())
        if train_cohort_rows != EXPECTED_EXACT_RATIONAL_TRAIN_ROWS:
            raise RuntimeError(
                f"{COHORT_NAME} train count drift: {train_cohort_rows} "
                f"!= {EXPECTED_EXACT_RATIONAL_TRAIN_ROWS}"
            )
        if test_cohort_rows != EXPECTED_EXACT_RATIONAL_TEST_ROWS:
            raise RuntimeError(
                f"{COHORT_NAME} test count drift: {test_cohort_rows} "
                f"!= {EXPECTED_EXACT_RATIONAL_TEST_ROWS}"
            )

    train_df, dev_df = split_raw_train(
        annotated_train,
        seed=seed,
        dev_ratio=dev_ratio,
    )
    test_df = annotated_test

    annotated_columns = [*ORIGINAL_COLUMNS, *COHORT_COLUMNS]
    recovered = pd.concat([train_df, dev_df], ignore_index=True)
    invariants = {
        "train_dev_disjoint": set(train_df["problem"]).isdisjoint(set(dev_df["problem"])),
        "recover_annotated_raw_train": _same_rows_unordered(
            recovered,
            annotated_train,
            annotated_columns,
        ),
        "raw_test_preserved": test_df[ORIGINAL_COLUMNS].equals(
            raw_test[ORIGINAL_COLUMNS].reset_index(drop=True)
        ),
        "cohort_metadata_exists": all(
            set(COHORT_COLUMNS).issubset(frame.columns)
            for frame in (train_df, dev_df, test_df)
        ),
    }
    if not all(invariants.values()):
        failed = [name for name, passed in invariants.items() if not passed]
        raise RuntimeError(f"Processed invariant failure: {failed}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path_out = output_dir / "train.parquet"
    dev_path_out = output_dir / "dev.parquet"
    test_path_out = output_dir / "test.parquet"
    manifest_path = output_dir / "manifest.json"

    train_df.to_parquet(train_path_out, index=False)
    dev_df.to_parquet(dev_path_out, index=False)
    test_df.to_parquet(test_path_out, index=False)

    cohort_manifest = {
        "name": COHORT_NAME,
        "definition": (
            "gold solution contains exactly one well-formed non-empty \\boxed{...}; "
            "boxed content is a signed integer, a/b, or \\frac{a}{b}; denominator != 0; "
            "canonicalized with fractions.Fraction"
        ),
        "source_train": _cohort_summary(annotated_train),
        "train": _cohort_summary(train_df),
        "dev": _cohort_summary(dev_df),
        "test": _cohort_summary(test_df),
    }
    manifest: dict[str, Any] = {
        "raw": {
            "train_rows": len(raw_train),
            "test_rows": len(raw_test),
        },
        "split": {
            "seed": seed,
            "dev_ratio": dev_ratio,
            "stratify": ["type", "level"],
        },
        "processed": {
            "train_rows": len(train_df),
            "dev_rows": len(dev_df),
            "test_rows": len(test_df),
        },
        "cohort": cohort_manifest,
        "invariants": invariants,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("Processed MATH data")
    print(f"- train: {train_path_out} ({len(train_df)} rows)")
    print(f"- dev:   {dev_path_out} ({len(dev_df)} rows)")
    print(f"- test:  {test_path_out} ({len(test_df)} rows)")
    for name, frame in (
        ("source train", annotated_train),
        ("train", train_df),
        ("dev", dev_df),
        ("test", test_df),
    ):
        eligible = int(frame["rational_eligible"].sum())
        print(f"- {COHORT_NAME} {name}: {eligible}/{len(frame)} ({eligible / len(frame):.2%})")
    print(f"- manifest: {manifest_path}")
