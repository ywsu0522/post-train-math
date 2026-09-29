from decimal import localcontext
from fractions import Fraction

import pytest

from posttrain_math.answers import (
    classify_boxed_format,
    classify_boxed_numeric_solution,
    extract_final_boxed,
    extract_final_boxed_numeric,
    normalize_boxed_numeric_gold_solution,
    parse_boxed_numeric,
    verify_boxed_numeric,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("42", Fraction(42, 1)),
        ("-17", Fraction(-17, 1)),
        ("0.5", Fraction(1, 2)),
        ("-0.125", Fraction(-1, 8)),
        ("0001.2500", Fraction(5, 4)),
        (r"50\%", Fraction(1, 2)),
        (r"-12.5\%", Fraction(-1, 8)),
        (r"0.1\%", Fraction(1, 1000)),
        ("-0.000", Fraction(0)),
        (r"-0\%", Fraction(0)),
        (" \t0.5\n", Fraction(1, 2)),
        (r"\frac{6}{8}", Fraction(3, 4)),
        (r"-\frac{6}{8}", Fraction(-3, 4)),
        (r"\frac{006}{008}", Fraction(3, 4)),
        (r"-\frac{0}{8}", Fraction(0)),
    ],
)
def test_parse_boxed_numeric(text: str, expected: Fraction) -> None:
    assert parse_boxed_numeric(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "+17",
        "6/8",
        "-6/8",
        "6/-8",
        ".5",
        "5.",
        "1e3",
        "NaN",
        "Infinity",
        "50%",
        r"50 \%",
        "1 2",
        "1\n2",
        "−2",
        "１２",
        r"\frac{-6}{8}",
        r"\frac{6}{-8}",
        r"\frac{-6}{-8}",
        r"+\frac{6}{8}",
        r"- \frac{6}{8}",
        r"\frac {6}{8}",
        r"\frac{ 6}{8}",
        r"\frac{6} {8}",
        r"\frac{6}{8 }",
        r"\frac{6}{8}\%",
        r"\dfrac{6}{8}",
        "{2}",
        "1,000",
        r"\sqrt{2}",
        r"\frac{x+1}{2}",
        r"\frac{2}{0}",
        "2 euros",
        "(1, 2)",
        "[0, 1]",
        r"\pi",
        "x = 2",
        "",
        None,
    ],
)
def test_parse_boxed_numeric_rejects_out_of_domain(text: str | None) -> None:
    assert parse_boxed_numeric(text) is None


@pytest.mark.parametrize("suffix,scale", [("", 1), (r"\%", 100)])
def test_decimal_parsing_preserves_all_digits(suffix: str, scale: int) -> None:
    digits = "123456789012345678901234567890123456789"
    with localcontext() as context:
        context.prec = 2
        assert parse_boxed_numeric(f"0.{digits}{suffix}") == Fraction(
            int(digits), 10 ** len(digits) * scale
        )
        assert context.prec == 2


def test_final_boxed_uses_last_marker_without_fallback() -> None:
    text = r"First \boxed{1}; final \boxed{\sqrt{2}}"
    assert extract_final_boxed(text) == r"\sqrt{2}"
    assert extract_final_boxed_numeric(text) is None


@pytest.mark.parametrize("final", [r"\boxed 2", r"\boxed", r"\boxed{2", r"\boxed{}"])
def test_malformed_final_box_does_not_fall_back(final: str) -> None:
    text = r"First \boxed{1}; final " + final
    assert extract_final_boxed(text) is None
    assert extract_final_boxed_numeric(text) is None


def test_gold_cohort_requires_exactly_one_boxed_marker() -> None:
    eligible = classify_boxed_numeric_solution(r"Work. \boxed{\frac{2}{4}}")
    assert eligible.eligible is True
    assert eligible.gt_boxed == r"\frac{2}{4}"
    assert eligible.numerator == 1
    assert eligible.denominator == 2

    multiple = classify_boxed_numeric_solution(r"\boxed{1} then \boxed{2}")
    assert multiple.eligible is False
    assert multiple.exclusion_reason == "multiple_boxed"

    # Count literal markers, including nested, escaped and malformed occurrences.
    for solution in (r"\boxed{\boxed{1}}", r"\\boxed{1} \boxed", r"\boxed{1} \boxed 2"):
        assert classify_boxed_numeric_solution(solution).exclusion_reason == "multiple_boxed"

    decimal = classify_boxed_numeric_solution("Work. \\boxed{ \t50\\%\n }")
    assert decimal.eligible
    assert decimal.gt_boxed == r"50\%"
    assert decimal.fraction == Fraction(1, 2)


def test_normalize_boxed_numeric_gold_solution_for_sft() -> None:
    raw = "Reasoning. \\boxed \n {  \\frac{2}{4}  } trailing."
    assert normalize_boxed_numeric_gold_solution(raw) == (
        r"Reasoning. \boxed{\frac{2}{4}} trailing."
    )

    with pytest.raises(ValueError, match="not eligible"):
        normalize_boxed_numeric_gold_solution(
            r"Reasoning. \boxed{\sqrt{2}}"
        )


def test_gold_cohort_exclusion_reasons() -> None:
    assert classify_boxed_numeric_solution("answer 2").exclusion_reason == "no_boxed"
    assert classify_boxed_numeric_solution(r"\boxed{}").exclusion_reason == "empty_boxed"
    assert classify_boxed_numeric_solution(r"\boxed 2").exclusion_reason == "malformed_boxed"
    assert (
        classify_boxed_numeric_solution(r"\boxed{\sqrt{2}}").exclusion_reason
        == "non_numeric_boxed"
    )
    assert (
        classify_boxed_numeric_solution(r"\boxed{\frac{2}{0}}").exclusion_reason
        == "zero_denominator"
    )


def test_boxed_format_classifier() -> None:
    assert classify_boxed_format(r"\boxed{2}") == "single_valid"
    assert classify_boxed_format(r"\boxed{1}\boxed{2}") == "multiple_boxed"
    assert classify_boxed_format(r"\boxed{}") == "empty_boxed"
    assert classify_boxed_format(r"\boxed 2") == "malformed_boxed"
    assert classify_boxed_format("answer 2") == "no_boxed"


@pytest.mark.parametrize("answer", [r"\frac{2}{4}", "0.5", r"50\%", "0.5000"])
def test_verify_boxed_numeric_canonicalizes_prediction(answer: str) -> None:
    assert verify_boxed_numeric(
        rf"Reasoning. Final: \boxed{{{answer}}}",
        numerator=1,
        denominator=2,
    )
    assert not verify_boxed_numeric(
        r"Reasoning. Final: \boxed{2/4}",
        numerator=1,
        denominator=2,
    )
