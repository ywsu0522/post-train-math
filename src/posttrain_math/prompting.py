from __future__ import annotations

import hashlib
from collections.abc import Callable

PromptFormatter = Callable[[str], str]
PROMPT_CONTRACT = "boxed-numeric-prompt-v1"
SYSTEM_PROMPT = r'''Solve the math problem and show your reasoning. Put your final answer in \boxed{...}.
Inside the box, output only an answer matching this grammar, without whitespace:
digits := [0-9]+
number := ["-"] digits ["." digits] ["\%"]
fraction := ["-"] "\frac{" digits "}{" digits "}"
answer := number | fraction
Use an optional leading minus only. Percentages require the literal \% suffix.
Do not use slash fractions, plus signs, scientific notation, units, or prose inside the box.'''


def format_boxed_prompt(problem: str) -> str:
    problem = problem.strip()
    if not problem:
        raise ValueError("problem must not be empty")
    # The pinned base tokenizer has no chat template. Use this exact text
    # serialization for base inference, completion-only SFT, and RL prompts.
    return f"System:\n{SYSTEM_PROMPT}\n\nProblem:\n{problem}\n\nSolution:\n"


PROMPT_FORMATTERS: dict[str, PromptFormatter] = {"boxed": format_boxed_prompt}
PROMPT_STRATEGIES = tuple(PROMPT_FORMATTERS)


def get_prompt_formatter(name: str) -> PromptFormatter:
    try:
        return PROMPT_FORMATTERS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown prompt strategy: {name}") from exc


def prompt_metadata() -> dict[str, str]:
    return {
        "contract": PROMPT_CONTRACT,
        "system_prompt": SYSTEM_PROMPT,
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "serialization": "System/Problem/Solution-v1",
    }
