"""Cohort filtering and checkpoint/parallelism contracts for L1–3 SFT."""

import pandas as pd
import pytest

from posttrain_math.cli import build_parser
from posttrain_math.distributed import resolve_gradient_accumulation
from posttrain_math.training import _load_split


def _fixture_split(tmp_path):
    data = pd.DataFrame(
        [
            {"problem": "a", "solution": "\\boxed{1}", "level": "Level 1", "numeric_eligible": True},
            {"problem": "b", "solution": "\\boxed{2}", "level": "Level 2", "numeric_eligible": True},
            {"problem": "c", "solution": "\\boxed{3}", "level": "Level 3", "numeric_eligible": True},
            {"problem": "d", "solution": "\\boxed{4}", "level": "Level 4", "numeric_eligible": True},
            {"problem": "e", "solution": "\\boxed{5}", "level": "Level 5", "numeric_eligible": True},
            {"problem": "f", "solution": "not-numeric", "level": "Level 1", "numeric_eligible": False},
        ]
    )
    for split in ("train", "dev"):
        data.to_parquet(tmp_path / f"{split}.parquet", index=False)


def test_filter_numeric_and_level_for_train_and_dev(tmp_path):
    _fixture_split(tmp_path)
    for split in ("train", "dev"):
        filtered = _load_split(tmp_path, split, levels=(1, 2, 3))
        assert filtered["problem"].tolist() == ["a", "b", "c"]
        assert filtered.index.tolist() == [0, 1, 2]
        assert _load_split(tmp_path, split)["problem"].tolist() == ["a", "b", "c", "d", "e"]


def test_bad_levels_rejected(tmp_path):
    _fixture_split(tmp_path)
    for levels in ((), (0,), (6,)):
        with pytest.raises(ValueError, match="levels must contain"):
            _load_split(tmp_path, "train", levels=levels)
    df = pd.read_parquet(tmp_path / "train.parquet")
    df.loc[df["level"].eq("Level 5"), "numeric_eligible"] = False
    df.to_parquet(tmp_path / "train.parquet", index=False)
    with pytest.raises(ValueError, match="empty"):
        _load_split(tmp_path, "train", levels=(5,))


def test_cli_sft_levels_epoch_and_inspect():
    args = build_parser().parse_args(
        ["train", "sft", "--levels", "1", "2", "3", "--epochs", "3", "--save-strategy", "epoch"]
    )
    assert args.levels == [1, 2, 3]
    assert args.epochs == 3.0
    assert args.save_strategy == "epoch"
    inspect_args = build_parser().parse_args(["train", "inspect", "--levels", "1", "2", "3"])
    assert inspect_args.levels == [1, 2, 3]
    eval_args = build_parser().parse_args(["eval", "--levels", "1", "2", "3"])
    assert eval_args.levels == [1, 2, 3]


def test_ddp_batch_invariance():
    for world_size, expected_accumulation in ((1, 8), (2, 4)):
        accumulation, effective_batch = resolve_gradient_accumulation(
            per_device_batch_size=2,
            world_size=world_size,
            gradient_accumulation=None,
            global_batch_size=16,
        )
        assert accumulation == expected_accumulation
        assert effective_batch == 16


def test_greedy_evaluation_filters_l123(tmp_path):
    from posttrain_math.evaluation import evaluate

    class FakeGenerator:
        def iter_generate(self, prompts, *, max_new_tokens, do_sample):
            assert len(prompts) == 2
            assert max_new_tokens == 32
            assert do_sample is False
            yield r"\boxed{1}"
            yield r"\boxed{2}"

        def count_tokens(self, texts):
            return [1] * len(texts)

        def metadata(self):
            return {"backend": "fake"}

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    frame = pd.DataFrame(
        {
            "problem": ["p1", "p2", "p4", "p5"],
            "solution": [r"\boxed{1}", r"\boxed{2}", r"\boxed{4}", r"\boxed{5}"],
            "type": ["Algebra"] * 4,
            "level": ["Level 1", "Level 2", "Level 4", "Level 5"],
            "gt_boxed": ["1", "2", "4", "5"],
            "gt_numerator": [1, 2, 4, 5],
            "gt_denominator": [1] * 4,
            "numeric_eligible": [True] * 4,
        }
    )
    frame.to_parquet(data_dir / "dev.parquet", index=False)
    metrics = evaluate(
        FakeGenerator(),
        prompt_formatter=lambda s: s,
        prompt_name="boxed",
        data_dir=data_dir,
        split="dev",
        output_dir=tmp_path / "results",
        max_new_tokens=32,
        levels=(1, 2, 3),
    )
    assert metrics["num_examples"] == 2
    assert metrics["num_correct"] == 2
    assert metrics["cohort"]["levels"] == [1, 2, 3]
    assert set(metrics["accuracy_by_level"]) == {"Level 1", "Level 2"}
