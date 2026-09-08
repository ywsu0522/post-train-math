from fractions import Fraction

import pytest

from posttrain_math.answers import (
    classify_boxed_format,
    classify_exact_rational_solution,
    extract_final_boxed,
    extract_final_boxed_rational,
    parse_exact_rational,
    verify_exact_rational,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("42", Fraction(42, 1)),
        ("-17", Fraction(-17, 1)),
        ("+17", Fraction(17, 1)),
        ("6/8", Fraction(3, 4)),
        ("-6/8", Fraction(-3, 4)),
        ("6/-8", Fraction(-3, 4)),
        (r"\frac{6}{8}", Fraction(3, 4)),
        (r"-\frac{6}{8}", Fraction(-3, 4)),
        (r"\frac{-6}{-8}", Fraction(3, 4)),
    ],
)
def test_parse_exact_rational(text: str, expected: Fraction) -> None:
    assert parse_exact_rational(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "0.5",
        r"\sqrt{2}",
        r"\frac{x+1}{2}",
        r"\frac{2}{0}",
        "2 euros",
        "(1, 2)",
        "[0, 1]",
        r"\pi",
        "x = 2",
        "",
    ],
)
def test_parse_exact_rational_rejects_out_of_domain(text: str) -> None:
    assert parse_exact_rational(text) is None


def test_final_boxed_uses_last_marker_without_fallback() -> None:
    text = r"First \boxed{1}; final \boxed{\sqrt{2}}"
    assert extract_final_boxed(text) == r"\sqrt{2}"
    assert extract_final_boxed_rational(text) is None


def test_malformed_final_box_does_not_fall_back() -> None:
    text = r"First \boxed{1}; final \boxed 2"
    assert extract_final_boxed(text) is None
    assert extract_final_boxed_rational(text) is None


def test_gold_cohort_requires_exactly_one_boxed_marker() -> None:
    eligible = classify_exact_rational_solution(r"Work. \boxed{\frac{2}{4}}")
    assert eligible.eligible is True
    assert eligible.gt_boxed == r"\frac{2}{4}"
    assert eligible.numerator == 1
    assert eligible.denominator == 2

    multiple = classify_exact_rational_solution(r"\boxed{1} then \boxed{2}")
    assert multiple.eligible is False
    assert multiple.exclusion_reason == "multiple_boxed"


def test_gold_cohort_exclusion_reasons() -> None:
    assert classify_exact_rational_solution("answer 2").exclusion_reason == "no_boxed"
    assert classify_exact_rational_solution(r"\boxed{}").exclusion_reason == "empty_boxed"
    assert classify_exact_rational_solution(r"\boxed 2").exclusion_reason == "malformed_boxed"
    assert (
        classify_exact_rational_solution(r"\boxed{\sqrt{2}}").exclusion_reason
        == "non_rational_boxed"
    )
    assert (
        classify_exact_rational_solution(r"\boxed{\frac{2}{0}}").exclusion_reason
        == "zero_denominator"
    )


def test_boxed_format_classifier() -> None:
    assert classify_boxed_format(r"\boxed{2}") == "single_valid"
    assert classify_boxed_format(r"\boxed{1}\boxed{2}") == "multiple_boxed"
    assert classify_boxed_format(r"\boxed{}") == "empty_boxed"
    assert classify_boxed_format(r"\boxed 2") == "malformed_boxed"
    assert classify_boxed_format("answer 2") == "no_boxed"


def test_verify_exact_rational_canonicalizes_prediction() -> None:
    assert verify_exact_rational(
        r"Reasoning. Final: \boxed{2/4}",
        numerator=1,
        denominator=2,
    )
    assert not verify_exact_rational(
        r"Reasoning. Final: \boxed{0.5}",
        numerator=1,
        denominator=2,
    )
