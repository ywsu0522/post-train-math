import pandas as pd

from posttrain_math.data import (
    ORIGINAL_COLUMNS,
    add_eval_eligibility,
    add_gt_boxed,
    split_raw_train,
)


def make_dataset() -> pd.DataFrame:
    rows = []

    for level in (
        "Level 1",
        "Level 2",
    ):
        for problem_type in (
            "Algebra",
            "Geometry",
        ):
            for index in range(10):
                rows.append(
                    {
                        "problem": (
                            f"{level} "
                            f"{problem_type} "
                            f"{index}"
                        ),
                        "solution": (
                            "\\boxed{"
                            f"{index}"
                            "}"
                        ),
                        "type":
                            problem_type,
                        "level":
                            level,
                    }
                )

    return pd.DataFrame(
        rows,
        columns=ORIGINAL_COLUMNS,
    )


def test_add_gt_boxed() -> None:
    df = add_gt_boxed(
        make_dataset()
    )

    assert "gt_boxed" in df.columns
    assert df.iloc[0]["gt_boxed"] == "0"


def test_add_eval_eligibility() -> None:
    df = make_dataset().iloc[:2].copy()
    df.loc[df.index[1], "solution"] = "no boxed answer"
    df = add_eval_eligibility(add_gt_boxed(df))

    assert df["eval_eligible"].tolist() == [True, False]


def test_split_is_deterministic() -> None:
    df = add_gt_boxed(
        make_dataset()
    )

    train1, dev1 = split_raw_train(
        df,
        seed=42,
        dev_ratio=0.2,
    )

    train2, dev2 = split_raw_train(
        df,
        seed=42,
        dev_ratio=0.2,
    )

    pd.testing.assert_frame_equal(
        train1,
        train2,
    )

    pd.testing.assert_frame_equal(
        dev1,
        dev2,
    )


def test_train_dev_disjoint() -> None:
    df = add_gt_boxed(
        make_dataset()
    )

    train, dev = split_raw_train(
        df,
        seed=42,
        dev_ratio=0.2,
    )

    assert set(
        train["problem"]
    ).isdisjoint(
        set(dev["problem"])
    )

def test_download_raw_datasets_materializes_local_parquets(tmp_path, monkeypatch) -> None:
    import pandas as pd

    import posttrain_math.data as data_module

    class FakeInfo:
        sha = "dataset-sha"

    class FakeApi:
        def dataset_info(self, repo_id: str, revision: str):
            assert repo_id == "org/math"
            assert revision == "main"
            return FakeInfo()

    class FakeSplit:
        def __init__(self, rows):
            self._rows = rows

        def to_pandas(self):
            return pd.DataFrame(self._rows)

    def fake_load_dataset(repo_id, config_name, revision):
        assert repo_id == "org/math"
        assert revision == "dataset-sha"
        row = {
            "problem": f"{config_name} problem",
            "solution": r"\\boxed{1}",
            "type": config_name,
            "level": "Level 1",
        }
        return {
            "train": FakeSplit([row]),
            "test": FakeSplit([row]),
        }

    monkeypatch.setattr(data_module, "HfApi", FakeApi)
    monkeypatch.setattr(data_module.hf_datasets, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(data_module, "EXPECTED_RAW_TRAIN_ROWS", 7)
    monkeypatch.setattr(data_module, "EXPECTED_RAW_TEST_ROWS", 7)

    train_path, test_path = data_module.download_raw_datasets(
        output_dir=tmp_path / "data",
        repo_id="org/math",
    )

    assert train_path.is_file()
    assert test_path.is_file()
    assert len(pd.read_parquet(train_path)) == 7
    assert len(pd.read_parquet(test_path)) == 7


def test_download_raw_datasets_reuses_complete_local_copy(tmp_path, monkeypatch) -> None:
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
