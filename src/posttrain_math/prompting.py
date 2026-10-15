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


def format_boxed_1shot_prompt(problem: str) -> str:
    """Fixed example and explicit boxed-answer instruction for L1–3 experiments.

    The example is a hand-written arithmetic question, not a MATH split item.
    Keep this exact formatter shared by SFT, evaluation and RL rollouts.
    """
    problem = _validate_problem(problem)
    return (
        "Follow the example below. Solve the problem step by step.\n"
        "Always end your solution with a final answer in \\boxed{...}.\n\n"
        "Example:\n"
        "Problem:\n"
        "What is 3/4 + 1/4?\n\n"
        "Solution:\n"
        "The denominators are equal, so we add the numerators.\n"
        "3/4 + 1/4 = 4/4 = 1.\n"
        "Final answer: \\boxed{1}\n\n"
        "Problem:\n"
        f"{problem}\n\n"
        "Solution:\n"
    )


PROMPT_FORMATTERS: dict[
    str,
    PromptFormatter,
] = {
    "plain": format_plain_prompt,
    "boxed": format_boxed_prompt,
    "boxed-cot": format_boxed_cot_prompt,
    "boxed-1shot": format_boxed_1shot_prompt,
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