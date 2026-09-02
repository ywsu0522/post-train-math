from posttrain_math.prompting import (
    PROMPT_STRATEGIES,
    format_boxed_cot_prompt,
    format_boxed_prompt,
    format_plain_prompt,
    get_prompt_formatter,
)


def test_plain_prompt() -> None:
    prompt = format_plain_prompt(
        "What is 1 + 1?"
    )

    assert "What is 1 + 1?" in prompt
    assert r"\boxed{...}" not in prompt
    assert "Think step-by-step" not in prompt


def test_boxed_prompt() -> None:
    prompt = format_boxed_prompt(
        "What is 1 + 1?"
    )

    assert "What is 1 + 1?" in prompt
    assert r"\boxed{...}" in prompt
    assert "Think step-by-step" not in prompt


def test_boxed_cot_prompt() -> None:
    prompt = format_boxed_cot_prompt(
        "What is 1 + 1?"
    )

    assert "What is 1 + 1?" in prompt
    assert r"\boxed{...}" in prompt
    assert "Think step-by-step" in prompt


def test_prompt_strategies_are_registered() -> None:
    assert set(PROMPT_STRATEGIES) == {
        "plain",
        "boxed",
        "boxed-cot",
    }


def test_get_prompt_formatter() -> None:
    assert (
        get_prompt_formatter("plain")
        is format_plain_prompt
    )

    assert (
        get_prompt_formatter("boxed")
        is format_boxed_prompt
    )

    assert (
        get_prompt_formatter("boxed-cot")
        is format_boxed_cot_prompt
    )
