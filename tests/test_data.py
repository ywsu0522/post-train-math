from pathlib import Path

import pandas as pd

from posttrain_math.data import (
    ORIGINAL_COLUMNS,
    annotate_rational_cohort,
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


def test_annotate_rational_cohort() -> None:
    df = pd.DataFrame(
        [
            {
                "problem": "a",
                "solution": r"\boxed{6/8}",
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
    result = annotate_rational_cohort(df)
    assert result["rational_eligible"].tolist() == [True, False]
    assert result.iloc[0]["gt_boxed"] == "6/8"
    assert int(result.iloc[0]["gt_numerator"]) == 3
    assert int(result.iloc[0]["gt_denominator"]) == 4
    assert result.iloc[1]["rational_exclusion"] == "non_rational_boxed"


def test_split_is_deterministic_and_disjoint() -> None:
    df = annotate_rational_cohort(make_dataset())
    train1, dev1 = split_raw_train(df, seed=42, dev_ratio=0.2)
    train2, dev2 = split_raw_train(df, seed=42, dev_ratio=0.2)
    pd.testing.assert_frame_equal(train1, train2)
    pd.testing.assert_frame_equal(dev1, dev2)
    assert set(train1["problem"]).isdisjoint(set(dev1["problem"]))


def test_download_raw_datasets_uses_two_consolidated_files(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import posttrain_math.data as data_module

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


def test_download_raw_datasets_reuses_complete_local_copy(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import posttrain_math.data as data_module

    output_dir = tmp_path / "data"
    output_dir.mkdir(parents=True)
    train_path = output_dir / "math_train.parquet"
    test_path = output_dir / "math_test.parquet"
    train_path.write_bytes(b"existing")
    test_path.write_bytes(b"existing")
    (output_dir / "download_manifest.json").write_text("{}", encoding="utf-8")

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
