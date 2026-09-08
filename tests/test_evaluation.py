from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd

from posttrain_math.evaluation import evaluate


class FakeGenerator:
    def iter_generate(
        self,
        prompts: list[str],
        *,
        max_new_tokens: int,
        do_sample: bool,
    ) -> Iterator[str]:
        assert len(prompts) == 2
        assert max_new_tokens == 128
        assert do_sample is False
        yield r"Answer: \boxed{2/2}"
        yield r"Answer: \boxed{0.5}"

    def generate(
        self,
        prompts: list[str],
        *,
        max_new_tokens: int,
        do_sample: bool,
    ) -> list[str]:
        return list(
            self.iter_generate(
                prompts,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
            )
        )

    def count_tokens(self, texts: list[str]) -> list[int]:
        return [len(text.split()) for text in texts]

    def metadata(self) -> dict[str, Any]:
        return {"backend": "fake"}


def fake_prompt(problem: str) -> str:
    return f"Problem: {problem}"


def test_evaluation_uses_fixed_rational_cohort_and_grouped_accuracy(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    df = pd.DataFrame(
        {
            "problem": ["1?", "1/2?", "unscorable"],
            "solution": [r"\boxed{1}", r"\boxed{1/2}", r"\boxed{\sqrt{2}}"],
            "type": ["Algebra", "Geometry", "Algebra"],
            "level": ["Level 1", "Level 2", "Level 5"],
            "gt_boxed": ["1", "1/2", r"\sqrt{2}"],
            "gt_numerator": pd.array([1, 1, None], dtype="Int64"),
            "gt_denominator": pd.array([1, 2, None], dtype="Int64"),
            "rational_eligible": [True, True, False],
            "rational_exclusion": [None, None, "non_rational_boxed"],
        }
    )
    df.to_parquet(data_dir / "dev.parquet", index=False)

    metrics = evaluate(
        FakeGenerator(),
        prompt_formatter=fake_prompt,
        prompt_name="fake",
        data_dir=data_dir,
        split="dev",
        output_dir=tmp_path / "run",
        max_new_tokens=128,
    )

    assert metrics["verifier"] == "exact-rational-v1"
    assert metrics["cohort"] == {
        "source_rows": 3,
        "eligible_rows": 2,
        "excluded_rows": 1,
        "evaluated_rows": 2,
        "limit": None,
    }
    assert metrics["num_examples"] == 2
    assert metrics["num_correct"] == 1
    assert metrics["accuracy"] == 0.5
    assert metrics["boxed_output_rate"] == 1.0
    assert metrics["rational_output_rate"] == 0.5
    assert metrics["accuracy_by_level"]["Level 1"]["accuracy"] == 1.0
    assert metrics["accuracy_by_level"]["Level 2"]["accuracy"] == 0.0
    assert metrics["accuracy_by_type"]["Algebra"]["accuracy"] == 1.0
    assert metrics["accuracy_by_type"]["Geometry"]["accuracy"] == 0.0
