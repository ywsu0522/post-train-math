import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from post_train_math.testing import (
    TEST_SAMPLE_IDS_SHA256,
    ids_sha256,
    problem_id,
    stratified_indices,
)


def test_proportional_sampling_is_reproducible_and_without_replacement():
    frame = pd.DataFrame(
        [
            {"problem": f"{subject}-{index}", "type": subject, "level": "Level 1"}
            for subject, count in (("a", 8), ("b", 6), ("c", 4), ("d", 2))
            for index in range(count)
        ]
    )
    first, strata = stratified_indices(frame, size=8, seed=42)
    assert stratified_indices(frame, size=8, seed=42) == (first, strata)
    assert len(set(first)) == 8 and first == sorted(first)
    assert [group["sample"] for group in strata] == [3, 2, 2, 1]
    assert stratified_indices(frame, size=8, seed=43)[0] != first
    with pytest.raises(ValueError, match="Duplicate"):
        stratified_indices(pd.concat([frame, frame]), size=8, seed=42)


def test_committed_512_questions_are_concrete_disjoint_and_hash_bound():
    root = Path(__file__).resolve().parents[1]
    folder = root / "cohort/testing-subset"
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    payload = (folder / "samples.jsonl").read_bytes()
    rows = [json.loads(line) for line in payload.decode("utf-8").strip().split("\n")]
    ids = [row["problem_id"] for row in rows]
    assert len(rows) == len(set(ids)) == 512
    assert manifest["sample_size"] == 512 and manifest["pool_size"] == 3686
    assert len(manifest["strata"]) == 35
    assert sum(group["sample"] for group in manifest["strata"]) == 512
    assert all(group["sample"] > 0 for group in manifest["strata"])
    assert hashlib.sha256(payload).hexdigest() == manifest["samples_sha256"]
    assert ids_sha256(ids) == TEST_SAMPLE_IDS_SHA256 == manifest["problem_ids_sha256"]
    assert (
        manifest["data_manifest_sha256"]
        == hashlib.sha256((root / "cohort/manifest.json").read_bytes()).hexdigest()
    )
    assert all(row["problem_id"] == problem_id(row["problem"]) for row in rows)
    indices = [row["test_cohort_index"] for row in rows]
    assert indices == sorted(set(indices))
    train = [
        json.loads(line)
        for line in (root / "cohort/training-subsets/olmo2/prompt_audit.jsonl")
        .read_text(encoding="utf-8")
        .strip()
        .split("\n")
    ]
    assert set(ids).isdisjoint({row["problem_id"] for row in train})
