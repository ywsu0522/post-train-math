import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from post_train_math.data import (
    ORIGINAL_COLUMNS,
    annotate_numeric_cohort,
    split_raw_train,
)


def make_dataset() -> pd.DataFrame:
    rows = []
    for level in ("Level 1", "Level 2"):
        for problem_type in ("Algebra", "Geometry"):
            for index in range(10):
                rows.append(
                    {
                        "problem": f"{level} {problem_type} {index}",
                        "solution": rf"\boxed{{{index}}}",
                        "type": problem_type,
                        "level": level,
                    }
                )
    return pd.DataFrame(rows, columns=ORIGINAL_COLUMNS)


@pytest.mark.parametrize("answer", [r"\frac{6}{8}", "6/8", "0.7500", r"75\%"])
def test_annotate_numeric_cohort(answer: str) -> None:
    df = pd.DataFrame(
        [
            {
                "problem": "a",
                "solution": rf"\boxed{{{answer}}}",
                "type": "Algebra",
                "level": "Level 1",
            },
            {
                "problem": "b",
                "solution": r"\boxed{\sqrt{2}}",
                "type": "Algebra",
                "level": "Level 1",
            },
        ],
        columns=ORIGINAL_COLUMNS,
    )
    result = annotate_numeric_cohort(df)
    assert result["numeric_eligible"].tolist() == [True, False]
    assert result.iloc[0]["gt_boxed"] == answer
    assert int(result.iloc[0]["gt_numerator"]) == 3
    assert int(result.iloc[0]["gt_denominator"]) == 4
    assert result.iloc[1]["numeric_exclusion"] == "non_numeric_boxed"


def test_prepare_numeric_metadata_and_cohort_audit(tmp_path: Path, monkeypatch) -> None:
    import post_train_math.data as data_module

    train = make_dataset()
    train["solution"] = [
        r"\boxed{12.5\%}",
        r"\boxed{1/8}",
        r"\boxed{x}",
        r"\boxed{1/8}",
    ] * 10
    test = make_dataset()
    test["problem"] = "test " + test["problem"]
    test["solution"] = [r"\boxed{\frac{2}{16}}", r"\boxed{\frac{1}{0}}"] * 20
    train_path, test_path = (
        tmp_path / "raw_train.parquet",
        tmp_path / "raw_test.parquet",
    )
    train.to_parquet(train_path, index=False)
    test.to_parquet(test_path, index=False)
    monkeypatch.setattr(data_module, "EXPECTED_RAW_TRAIN_ROWS", len(train))
    monkeypatch.setattr(data_module, "EXPECTED_RAW_TEST_ROWS", len(test))
    monkeypatch.setattr(
        data_module,
        "MATH_TRAIN_SHA256",
        hashlib.sha256(train_path.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        data_module,
        "MATH_TEST_SHA256",
        hashlib.sha256(test_path.read_bytes()).hexdigest(),
    )

    def split_before_annotation(frame, **kwargs):
        assert list(frame.columns) == ORIGINAL_COLUMNS
        assert len(frame) == len(train)
        return split_raw_train(frame, **kwargs)

    monkeypatch.setattr(data_module, "split_raw_train", split_before_annotation)
    output = tmp_path / "processed"
    data_module.prepare_datasets(train_path, test_path, output)

    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["source"]["sha256"]["train"] == data_module.MATH_TRAIN_SHA256
    assert (
        manifest["processed_sha256"]["dev"]
        == hashlib.sha256((output / "dev.parquet").read_bytes()).hexdigest()
    )
    assert all(manifest["invariants"].values())
    cohort = manifest["cohort"]
    assert cohort["name"] == "boxed-numeric"
    assert cohort["source_train"]["eligible_rows"] == 30
    assert cohort["test"]["eligible_rows"] == 20
    assert cohort["combined"]["source_rows"] == 80
    assert cohort["combined"]["eligible_rows"] == 50
    assert cohort["combined"]["retention"] == 0.625
    assert cohort["combined"]["exclusion_counts"] == {
        "non_numeric_boxed": 10,
        "zero_denominator": 20,
    }
    assert cohort["train"]["eligible_rows"] + cohort["dev"]["eligible_rows"] == 30
    expected_train, expected_dev = split_raw_train(train, seed=42, dev_ratio=0.1)
    expected = {"train": expected_train, "dev": expected_dev, "test": test}
    assert manifest["pipeline"] == data_module.PIPELINE
    for split in ("train", "dev", "test"):
        frame = pd.read_parquet(output / f"{split}.parquet")
        assert set(frame.columns) == set(ORIGINAL_COLUMNS + data_module.COHORT_COLUMNS)
        pd.testing.assert_frame_equal(
            frame[ORIGINAL_COLUMNS],
            expected[split][ORIGINAL_COLUMNS].reset_index(drop=True),
        )
        eligible = frame[frame["numeric_eligible"]]
        assert eligible["gt_numerator"].eq(1).all()
        assert eligible["gt_denominator"].eq(8).all()
        assert eligible["numeric_exclusion"].isna().all()


def test_split_is_deterministic_and_disjoint() -> None:
    df = annotate_numeric_cohort(make_dataset())
    train1, dev1 = split_raw_train(df, seed=42, dev_ratio=0.2)
    train2, dev2 = split_raw_train(df, seed=42, dev_ratio=0.2)
    pd.testing.assert_frame_equal(train1, train2)
    pd.testing.assert_frame_equal(dev1, dev2)
    assert set(train1["problem"]).isdisjoint(set(dev1["problem"]))


def test_download_raw_datasets_uses_two_consolidated_files(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import post_train_math.data as data_module

    class FakeInfo:
        sha = "dataset-sha"

    class FakeApi:
        def dataset_info(self, repo_id: str, revision: str):
            assert repo_id == "org/math"
            assert revision == data_module.MATH_DATASET_REVISION
            return FakeInfo()

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    train_source = source_dir / "train.parquet"
    test_source = source_dir / "test.parquet"

    tiny_train = make_dataset().iloc[:4]
    tiny_test = make_dataset().iloc[4:7]
    tiny_train.to_parquet(train_source, index=False)
    tiny_test.to_parquet(test_source, index=False)

    calls: list[str] = []

    def fake_hf_hub_download(*, repo_id, filename, repo_type, revision):
        assert repo_id == "org/math"
        assert repo_type == "dataset"
        assert revision == "dataset-sha"
        calls.append(filename)
        if filename == data_module.MATH_TRAIN_FILE:
            return str(train_source)
        if filename == data_module.MATH_TEST_FILE:
            return str(test_source)
        raise AssertionError(filename)

    monkeypatch.setattr(data_module, "HfApi", FakeApi)
    monkeypatch.setattr(data_module, "hf_hub_download", fake_hf_hub_download)
    monkeypatch.setattr(data_module, "EXPECTED_RAW_TRAIN_ROWS", len(tiny_train))
    monkeypatch.setattr(data_module, "EXPECTED_RAW_TEST_ROWS", len(tiny_test))

    train_path, test_path = data_module.download_raw_datasets(
        output_dir=tmp_path / "data",
        repo_id="org/math",
    )

    assert calls == [data_module.MATH_TRAIN_FILE, data_module.MATH_TEST_FILE]
    assert len(pd.read_parquet(train_path)) == len(tiny_train)
    assert len(pd.read_parquet(test_path)) == len(tiny_test)


def test_download_raw_datasets_reuses_verified_local_copy(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import post_train_math.data as data_module

    output_dir = tmp_path / "data"
    output_dir.mkdir(parents=True)

    train_path = output_dir / "math_train.parquet"
    test_path = output_dir / "math_test.parquet"
    manifest_path = output_dir / "download_manifest.json"

    train_bytes = b"existing train"
    test_bytes = b"existing test"
    train_path.write_bytes(train_bytes)
    test_path.write_bytes(test_bytes)

    manifest = {
        "repo_id": "org/math",
        "requested_revision": data_module.MATH_DATASET_REVISION,
        "resolved_commit": "dataset-sha",
        "source_files": {
            "train": data_module.MATH_TRAIN_FILE,
            "test": data_module.MATH_TEST_FILE,
        },
        "sha256": {
            "train": hashlib.sha256(train_bytes).hexdigest(),
            "test": hashlib.sha256(test_bytes).hexdigest(),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    class FailApi:
        def __init__(self, *args, **kwargs):
            raise AssertionError("network lookup should not run")

    monkeypatch.setattr(data_module, "HfApi", FailApi)

    got_train, got_test = data_module.download_raw_datasets(
        output_dir=output_dir,
        repo_id="org/math",
    )

    assert got_train == train_path
    assert got_test == test_path


def test_download_raw_datasets_rejects_stale_local_revision(
    tmp_path: Path,
) -> None:
    import post_train_math.data as data_module

    output_dir = tmp_path / "data"
    output_dir.mkdir(parents=True)

    train_path = output_dir / "math_train.parquet"
    test_path = output_dir / "math_test.parquet"
    manifest_path = output_dir / "download_manifest.json"

    train_bytes = b"existing train"
    test_bytes = b"existing test"
    train_path.write_bytes(train_bytes)
    test_path.write_bytes(test_bytes)

    manifest = {
        "repo_id": "org/math",
        "requested_revision": "old-revision",
        "resolved_commit": "old-commit",
        "source_files": {
            "train": data_module.MATH_TRAIN_FILE,
            "test": data_module.MATH_TEST_FILE,
        },
        "sha256": {
            "train": hashlib.sha256(train_bytes).hexdigest(),
            "test": hashlib.sha256(test_bytes).hexdigest(),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeError,
        match="does not match the requested source",
    ):
        data_module.download_raw_datasets(
            output_dir=output_dir,
            repo_id="org/math",
            revision="new-revision",
        )


def test_download_raw_datasets_rejects_corrupted_local_file(
    tmp_path: Path,
) -> None:
    import post_train_math.data as data_module

    output_dir = tmp_path / "data"
    output_dir.mkdir(parents=True)

    train_path = output_dir / "math_train.parquet"
    test_path = output_dir / "math_test.parquet"
    manifest_path = output_dir / "download_manifest.json"

    original_train = b"original train"
    test_bytes = b"original test"

    train_path.write_bytes(original_train)
    test_path.write_bytes(test_bytes)

    manifest = {
        "repo_id": "org/math",
        "requested_revision": data_module.MATH_DATASET_REVISION,
        "resolved_commit": "dataset-sha",
        "source_files": {
            "train": data_module.MATH_TRAIN_FILE,
            "test": data_module.MATH_TEST_FILE,
        },
        "sha256": {
            "train": hashlib.sha256(original_train).hexdigest(),
            "test": hashlib.sha256(test_bytes).hexdigest(),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    # Simulate accidental modification after download.
    train_path.write_bytes(b"corrupted train")

    with pytest.raises(
        RuntimeError,
        match="train SHA256 mismatch",
    ):
        data_module.download_raw_datasets(
            output_dir=output_dir,
            repo_id="org/math",
        )


@pytest.mark.parametrize("seed,ratio", [(41, 0.1), (42, 0.2)])
def test_prepare_cannot_change_frozen_split(tmp_path, seed, ratio):
    from post_train_math.data import prepare_datasets

    with pytest.raises(ValueError, match="frozen split"):
        prepare_datasets(
            tmp_path / "train",
            tmp_path / "test",
            tmp_path / "out",
            seed=seed,
            dev_ratio=ratio,
        )
    assert not (tmp_path / "out").exists()


def test_filtering_does_not_move_ineligible_problems_between_splits():
    from post_train_math.data import annotate_numeric_cohort

    raw = make_dataset()
    raw.loc[raw.index % 3 == 0, "solution"] = r"\boxed{x}"
    raw_train, raw_dev = split_raw_train(raw, seed=42, dev_ratio=0.1)
    train = annotate_numeric_cohort(raw_train)
    dev = annotate_numeric_cohort(raw_dev)
    assert len(train) == 36 and len(dev) == 4
    assert train["problem"].tolist() == raw_train["problem"].tolist()
    assert dev["problem"].tolist() == raw_dev["problem"].tolist()
    assert set(train.loc[train.numeric_eligible, "problem"]).isdisjoint(
        set(dev.loc[dev.numeric_eligible, "problem"])
    )


def test_committed_manifest_freezes_membership_and_gold_rule():
    from post_train_math import data

    manifest = json.loads(
        (Path(__file__).resolve().parents[1] / "cohort/manifest.json").read_text()
    )
    assert manifest["source"] == {
        "repo_id": data.MATH_DATASET_REPO,
        "revision": data.MATH_DATASET_REVISION,
        "sha256": {"train": data.MATH_TRAIN_SHA256, "test": data.MATH_TEST_SHA256},
    }
    assert manifest["processed_sha256"] == data.FROZEN_PROCESSED_SHA256
    assert manifest["pipeline"] == data.PIPELINE
    assert manifest["split"] == {
        "seed": 42,
        "dev_ratio": 0.1,
        "stratify": ["type", "level"],
    }
    assert manifest["processed"] == {
        "train_rows": 6750,
        "dev_rows": 750,
        "test_rows": 5000,
    }
    assert manifest["cohort"]["definition"] == data.COHORT_DEFINITION
    assert {
        split: manifest["cohort"][split]["eligible_rows"]
        for split in ("train", "dev", "test")
    } == {
        "train": 5029,
        "dev": 560,
        "test": 3686,
    }
