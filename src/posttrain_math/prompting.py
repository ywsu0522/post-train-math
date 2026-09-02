from __future__ import annotations

from collections.abc import Callable

PromptFormatter = Callable[[str], str]


def _validate_problem(
    problem: str,
) -> str:
    problem = problem.strip()

    if not problem:
        raise ValueError(
            "problem must not be empty"
        )

    return problem


def format_plain_prompt(
    problem: str,
) -> str:
    problem = _validate_problem(problem)

    return (
        f"Problem:\n{problem}\n\n"
        "Solution:\n"
    )


def format_boxed_prompt(
    problem: str,
) -> str:
    problem = _validate_problem(problem)

    return (
        f"Problem:\n{problem}\n\n"
        "Solve the problem. "
        "Put the final answer in "
        "\\boxed{...}.\n\n"
        "Solution:\n"
    )


def format_boxed_cot_prompt(
    problem: str,
) -> str:
    problem = _validate_problem(problem)

    return (
        f"Problem:\n{problem}\n\n"
        "Think step-by-step. "
        "Put the final answer in "
        "\\boxed{...}.\n\n"
        "Solution:\n"
    )


PROMPT_FORMATTERS: dict[
    str,
    PromptFormatter,
] = {
    "plain": format_plain_prompt,
    "boxed": format_boxed_prompt,
    "boxed-cot": format_boxed_cot_prompt,
}


PROMPT_STRATEGIES = tuple(
    PROMPT_FORMATTERS
)


def get_prompt_formatter(
    name: str,
) -> PromptFormatter:
    try:
        return PROMPT_FORMATTERS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown prompt strategy: {name}"
        ) from exc