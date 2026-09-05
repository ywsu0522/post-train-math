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
        yield r"Answer: \boxed{2}"
        yield r"Answer: \boxed{5}"

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


def test_evaluation_uses_fixed_eligible_cohort_and_grouped_accuracy(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    df = pd.DataFrame(
        {
            "problem": ["1 + 1?", "2 + 2?", "unscorable"],
            "solution": [r"\boxed{2}", r"\boxed{4}", "no boxed answer"],
            "type": ["Algebra", "Geometry", "Algebra"],
            "level": ["Level 1", "Level 2", "Level 5"],
            "gt_boxed": ["2", "4", None],
            "eval_eligible": [True, True, False],
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
    assert metrics["parseable_output_rate"] == 1.0

    assert metrics["accuracy_by_level"]["Level 1"] == {
        "correct": 1,
        "n": 1,
        "accuracy": 1.0,
    }
    assert metrics["accuracy_by_level"]["Level 2"] == {
        "correct": 0,
        "n": 1,
        "accuracy": 0.0,
    }
    assert metrics["accuracy_by_type"]["Algebra"] == {
        "correct": 1,
        "n": 1,
        "accuracy": 1.0,
    }
    assert metrics["accuracy_by_type"]["Geometry"] == {
        "correct": 0,
        "n": 1,
        "accuracy": 0.0,
    }
