from __future__ import annotations

from fractions import Fraction
from typing import Any

from posttrain_math.answers import extract_final_boxed_rational


def completion_text(completion: Any) -> str:
    """Normalize TRL completion payloads to plain text for deterministic scoring."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts: list[str] = []
        for item in completion:
            if isinstance(item, dict) and "content" in item:
                parts.append(str(item["content"]))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    if isinstance(completion, dict) and "content" in completion:
        return str(completion["content"])
    return str(completion)


def score_exact_rational_completion(
    completion: Any,
    numerator: int,
    denominator: int,
) -> float:
    """Return the exact-rational-v1 binary terminal reward for one completion."""
    denominator = int(denominator)
    if denominator == 0:
        raise RuntimeError("RL dataset contains a zero gold denominator.")
    gold = Fraction(int(numerator), denominator)
    prediction = extract_final_boxed_rational(completion_text(completion))
    return float(prediction == gold)


def make_exact_rational_reward():
    """Build the callable reward shared by all exact-rational-v1 RL backends."""

    def exact_rational_reward(
        completions,
        gt_numerator,
        gt_denominator,
        **kwargs,
    ) -> list[float]:
        del kwargs
        return [
            score_exact_rational_completion(completion, numerator, denominator)
            for completion, numerator, denominator in zip(
                completions,
                gt_numerator,
                gt_denominator,
                strict=True,
            )
        ]

    exact_rational_reward.__name__ = "exact_rational_reward"
    return exact_rational_reward
