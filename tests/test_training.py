import os

import pytest

from posttrain_math.training import (
    IGNORE_INDEX,
    _load_split,
    _resolve_precision,
    build_sft_config,
    encode_sft_example,
)


class FakeTokenizer:
    eos_token_id = 999

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
    ):
        del add_special_tokens

        return {
            "input_ids": [
                ord(char)
                for char in text
            ]
        }


def fake_prompt(
    problem: str,
) -> str:
    return (
        f"Problem: {problem}\n"
        "Solution:\n"
    )


def test_completion_only_labels() -> None:
    tokenizer = FakeTokenizer()

    example = encode_sft_example(
        tokenizer,
        row_id=0,
        problem="1+1?",
        solution="2",
        prompt_formatter=fake_prompt,
    )

    assert all(
        label == IGNORE_INDEX
        for label
        in example.labels[
            : example.prompt_length
        ]
    )

    assert (
        example.labels[
            example.prompt_length :
        ]
        == example.input_ids[
            example.prompt_length :
        ]
    )


def test_eos_is_supervised() -> None:
    tokenizer = FakeTokenizer()

    example = encode_sft_example(
        tokenizer,
        row_id=0,
        problem="1+1?",
        solution="2",
        prompt_formatter=fake_prompt,
    )

    assert (
        example.input_ids[-1]
        == tokenizer.eos_token_id
    )

    assert (
        example.labels[-1]
        == tokenizer.eos_token_id
    )


def test_prompt_has_no_supervised_position() -> None:
    tokenizer = FakeTokenizer()

    example = encode_sft_example(
        tokenizer,
        row_id=0,
        problem="abc",
        solution="answer",
        prompt_formatter=fake_prompt,
    )

    supervised_prompt = sum(
        label != IGNORE_INDEX
        for label
        in example.labels[
            : example.prompt_length
        ]
    )

    assert supervised_prompt == 0

def test_precision_dtype_mapping() -> None:
    import torch

    from posttrain_math.training import _dtype_for_precision

    assert _dtype_for_precision("bf16") is torch.bfloat16
    assert _dtype_for_precision("fp16") is torch.float16
    assert _dtype_for_precision("fp32") is torch.float32


def test_training_uses_headless_matplotlib_backend() -> None:
    import matplotlib

    assert os.environ["MPLBACKEND"] == "Agg"
    assert matplotlib.get_backend().lower() == "agg"


def test_auto_precision_requires_native_bf16(monkeypatch) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "is_bf16_supported",
        lambda *, including_emulation=True: including_emulation,
    )

    assert _resolve_precision("auto") == "fp16"
    with pytest.raises(RuntimeError, match="native BF16"):
        _resolve_precision("bf16")


def test_lora_config_defaults_are_explicit() -> None:
    from posttrain_math.training import build_lora_config

    config = build_lora_config(
        r=16,
        alpha=32,
        dropout=0.05,
        target_modules="all-linear",
    )

    assert config.r == 16
    assert config.lora_alpha == 32
    assert config.lora_dropout == 0.05
    assert config.target_modules == "all-linear"
    assert config.bias == "none"


def test_sft_config_translates_warmup_ratio(tmp_path) -> None:
    config = build_sft_config(
        output_dir=str(tmp_path),
        warmup_ratio=0.03,
        bf16=False,
        fp16=False,
        use_cpu=True,
    )

    assert config.warmup_steps == 0.03
    assert config.get_warmup_steps(100) == 3


@pytest.mark.parametrize("split", ["train", "dev"])
def test_sft_split_is_always_numeric_eligible(tmp_path, split) -> None:
    import pandas as pd

    df = pd.DataFrame(
        {
            "problem": [
                "eligible",
                "ineligible",
                "unknown",
            ],
            "solution": [
                r"\boxed{2}",
                r"\boxed{x}",
                r"\boxed{3}",
            ],
            "numeric_eligible": [
                True,
                False,
                None,
            ],
        }
    )

    df.to_parquet(
        tmp_path / f"{split}.parquet",
        index=False,
    )

    loaded = _load_split(
        tmp_path,
        split,
    )

    assert len(loaded) == 1
    assert loaded.iloc[0]["problem"] == "eligible"
    assert loaded["numeric_eligible"].eq(True).all()


@pytest.mark.parametrize("split", ["train", "dev"])
@pytest.mark.parametrize("eligibility", [None, [False, None]])
def test_sft_split_rejects_missing_or_empty_numeric_cohort(tmp_path, split, eligibility) -> None:
    import pandas as pd

    frame = pd.DataFrame({"problem": ["a", "b"], "solution": ["x", "y"]})
    if eligibility is not None:
        frame["numeric_eligible"] = eligibility
    frame.to_parquet(tmp_path / f"{split}.parquet", index=False)
    with pytest.raises(ValueError, match="missing columns.*numeric_eligible|cohort is empty"):
        _load_split(tmp_path, split)
