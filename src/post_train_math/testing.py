"""One frozen, model-independent sample of the original boxed-numeric test set."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

TEST_SAMPLE_SIZE = 512
TEST_SAMPLE_SEED = 42
TEST_POOL_SIZE = 3686
TEST_SAMPLE_IDS_SHA256 = (
    "174d7a6241e33c11edf799b8dd5dbfa73b9e75e539516cdd41531532f04922d6"
)
SAMPLING_RULE = (
    "proportional allocation over (type, level) using largest remainders; "
    "within each stratum rank SHA256(seed + NUL + problem_id); "
    "return selected rows in original eligible-test order"
)


def problem_id(problem: str) -> str:
    return "math-" + hashlib.sha256(problem.encode("utf-8")).hexdigest()


def ids_sha256(ids: list[str]) -> str:
    return hashlib.sha256(("\n".join(ids) + "\n").encode("utf-8")).hexdigest()


def stratified_indices(frame, *, size: int, seed: int) -> tuple[list[int], list[dict]]:
    """Allocate proportionally without replacement, with deterministic tie breaks."""
    if not 0 < size <= len(frame):
        raise ValueError("Sample size must be positive and no larger than its pool")
    groups = defaultdict(list)
    ids = [problem_id(str(value)) for value in frame["problem"]]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate testing problem IDs")
    for index, row in enumerate(frame.itertuples(index=False)):
        groups[(str(row.type), str(row.level))].append(index)
    allocations = {key: size * len(rows) // len(frame) for key, rows in groups.items()}
    remainder_order = sorted(
        groups,
        key=lambda key: (-(size * len(groups[key]) % len(frame)), key),
    )
    for key in remainder_order[: size - sum(allocations.values())]:
        allocations[key] += 1
    selected, strata = [], []
    for key in sorted(groups):
        ranked = sorted(
            groups[key],
            key=lambda index: (
                hashlib.sha256(f"{seed}\0{ids[index]}".encode()).digest(),
                ids[index],
            ),
        )
        selected.extend(ranked[: allocations[key]])
        strata.append(
            {
                "type": key[0],
                "level": key[1],
                "pool": len(ranked),
                "sample": allocations[key],
            }
        )
    return sorted(selected), strata


def select_testing_set(frame, name: str = "fixed512"):
    """Reconstruct the same frozen sample on every machine and for every model."""
    if name == "full":
        indices, strata = list(range(len(frame))), None
    elif name == "fixed512":
        if len(frame) != TEST_POOL_SIZE:
            raise ValueError(f"Expected the frozen {TEST_POOL_SIZE}-question test pool")
        indices, strata = stratified_indices(
            frame, size=TEST_SAMPLE_SIZE, seed=TEST_SAMPLE_SEED
        )
    else:
        raise ValueError(f"Unknown testing set: {name}")
    selected = frame.iloc[indices].copy().reset_index(drop=True)
    selected["test_cohort_index"] = indices
    selected["problem_id"] = [problem_id(str(value)) for value in selected["problem"]]
    digest = ids_sha256(selected["problem_id"].tolist())
    if name == "fixed512" and digest != TEST_SAMPLE_IDS_SHA256:
        raise ValueError("Testing sample differs from the frozen 512 problem IDs")
    return selected, {
        "name": name,
        "pool_size": len(frame),
        "sample_size": len(selected),
        "problem_ids_sha256": digest,
        "seed": TEST_SAMPLE_SEED if name == "fixed512" else None,
        "sampling_rule": SAMPLING_RULE
        if name == "fixed512"
        else "all eligible original test rows",
        "strata": strata,
    }


def write_testing_sample(data_dir: Path, output_dir: Path) -> dict:
    """Materialize the audited questions; runtime selection verifies their ID hash."""
    from post_train_math.data import load_cohort, validate_processed_manifest

    _, parent_hash = validate_processed_manifest(data_dir)
    frame, definition = select_testing_set(load_cohort(data_dir, "test"))
    records = [
        {
            "problem_id": row.problem_id,
            "test_cohort_index": int(row.test_cohort_index),
            "problem": row.problem,
            "type": row.type,
            "level": row.level,
            "gt_numerator": int(row.gt_numerator),
            "gt_denominator": int(row.gt_denominator),
        }
        for row in frame.itertuples(index=False)
    ]
    payload = b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        for row in records
    )
    definition = {
        **definition,
        "data_manifest_sha256": parent_hash,
        "samples_sha256": hashlib.sha256(payload).hexdigest(),
        "accuracy_aggregation": "unweighted fraction correct over the fixed 512 questions",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "samples.jsonl").write_bytes(payload)
    (output_dir / "manifest.json").write_bytes(
        (
            json.dumps(definition, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        ).encode("utf-8")
    )
    return definition
