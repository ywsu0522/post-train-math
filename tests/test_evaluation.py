from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd

from posttrain_math.evaluation import (
    evaluate,
)


class FakeGenerator:
    def iter_generate(
        self,
        prompts: list[str],
        *,
        max_new_tokens: int,
        do_sample: bool,
    ) -> Iterator[str]:
        outputs = [
            r"Answer: \boxed{2}",
            r"Answer: \boxed{5}",
        ]

        yield from outputs

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
                max_new_tokens=(
                    max_new_tokens
                ),
                do_sample=do_sample,
            )
        )

    def count_tokens(
        self,
        texts: list[str],
    ) -> list[int]:
        return [
            len(text.split())
            for text in texts
        ]

    def metadata(
        self,
    ) -> dict[str, Any]:
        return {
            "backend": "fake",
        }


def fake_prompt(
    problem: str,
) -> str:
    return (
        f"Problem: {problem}"
    )


def test_evaluation(
    tmp_path: Path,
) -> None:
    data_dir = (
        tmp_path / "data"
    )

    data_dir.mkdir()

    df = pd.DataFrame(
        {
            "problem": [
                "1 + 1?",
                "2 + 2?",
            ],
            "solution": [
                r"\boxed{2}",
                r"\boxed{4}",
            ],
            "type": [
                "Algebra",
                "Algebra",
            ],
            "level": [
                "Level 1",
                "Level 1",
            ],
            "gt_boxed": [
                "2",
                "4",
            ],
        }
    )

    df.to_parquet(
        data_dir
        / "dev.parquet",
        index=False,
    )

    metrics = evaluate(
        FakeGenerator(),
        prompt_formatter=(
            fake_prompt
        ),
        prompt_name="fake",
        data_dir=data_dir,
        split="dev",
        output_dir=(
            tmp_path / "run"
        ),
        max_new_tokens=128,
    )

    assert (
        metrics["num_examples"]
        == 2
    )

    assert (
        metrics["num_correct"]
        == 1
    )

    assert (
        metrics["accuracy"]
        == 0.5
    )
