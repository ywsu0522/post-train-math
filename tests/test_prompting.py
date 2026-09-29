import pytest

from posttrain_math.prompting import (
    PROMPT_STRATEGIES,
    SYSTEM_PROMPT,
    format_boxed_prompt,
    get_prompt_formatter,
    prompt_metadata,
)


def test_shared_system_prompt_and_serialization() -> None:
    prompt = format_boxed_prompt(" What is 1 + 1? ")
    assert prompt == f"System:\n{SYSTEM_PROMPT}\n\nProblem:\nWhat is 1 + 1?\n\nSolution:\n"
    assert r"\boxed{...}" in prompt
    assert 'answer := number | fraction' in prompt
    assert r'number := ["-"] digits ["." digits] ["\%"]' in prompt
    assert r'fraction := ["-"] "\frac{" digits "}{" digits "}"' in prompt
    assert set(PROMPT_STRATEGIES) == {"boxed"}
    assert get_prompt_formatter("boxed") is format_boxed_prompt
    assert prompt_metadata()["system_prompt"] == SYSTEM_PROMPT


@pytest.mark.parametrize("name", ["plain", "boxed-cot", "unknown"])
def test_alternative_prompt_strategies_are_rejected(name) -> None:
    with pytest.raises(ValueError, match="Unknown prompt"):
        get_prompt_formatter(name)


def test_empty_problem_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        format_boxed_prompt("  ")
