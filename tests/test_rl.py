from pathlib import Path

import pandas as pd

from posttrain_math.data import annotate_numeric_cohort
from posttrain_math.rl import make_boxed_numeric_reward
from posttrain_math.rl_common import build_boxed_numeric_rl_dataset


def test_boxed_numeric_reward_is_binary_and_canonical() -> None:
    reward = make_boxed_numeric_reward()
    values = reward(
        [
            r"Final: \boxed{\frac{2}{4}}",
            r"Final: \boxed{\frac{3}{4}}",
            r"Final: \boxed{0.5}",
            "No boxed answer",
            r"Final: \boxed{50\%}",
            r"Final: \boxed{2/4}",
        ],
        [1] * 6,
        [2] * 6,
    )
    assert values == [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]


def test_final_boxed_marker_controls_reward() -> None:
    reward = make_boxed_numeric_reward()
    values = reward(
        [
            r"First \boxed{0.5}; final \boxed{\frac{3}{4}}",
            r"First \boxed{0.5}; final \boxed{\frac{1}{0}}",
            r"First \boxed{0.5}; final \boxed 2",
            r"First \boxed{0.5}; final \boxed{2/4}",
            r"First \boxed{2}; final \boxed{50\%}",
        ],
        [1] * 5,
        [2] * 5,
    )
    assert values == [0.0, 0.0, 0.0, 0.0, 1.0]


def test_rl_dataset_uses_numeric_cohort_and_canonical_gold(tmp_path: Path) -> None:
    frame = annotate_numeric_cohort(pd.DataFrame({
        "problem": ["a", "b", "c", "d"],
        "solution": [r"\boxed{50\%}", r"\boxed{0.5}", r"\boxed{\frac{2}{4}}", r"\boxed{1/2}"],
        "type": ["Algebra"] * 4,
        "level": ["Level 1"] * 4,
    }))
    frame.to_parquet(tmp_path / "train.parquet", index=False)

    def tokenizer(text, *, add_special_tokens):
        assert not add_special_tokens
        return {"input_ids": list(range(len(text)))}

    dataset, stats = build_boxed_numeric_rl_dataset(
        data_dir=tmp_path,
        tokenizer=tokenizer,
        prompt_name="boxed",
        max_prompt_length=1000,
        limit_prompts=None,
    )
    assert stats["source_rows"] == 4
    assert stats["cohort_rows"] == stats["selected_rows"] == 3
    assert stats["ineligible_rows"] == 1
    assert list(dataset["gt_numerator"]) == [1, 1, 1]
    assert list(dataset["gt_denominator"]) == [2, 2, 2]
