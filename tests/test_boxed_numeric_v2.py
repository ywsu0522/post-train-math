"""Version-2 slash fraction grammar: strict parsing, cohort, reward, provenance."""

import json
from fractions import Fraction

import pytest

from posttrain_math.answers import (
    classify_boxed_numeric_solution,
    extract_final_boxed_numeric,
    parse_boxed_numeric,
    verify_boxed_numeric,
)
from posttrain_math.data import COHORT_NAME, assert_current_cohort


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("9/4", Fraction(9, 4)),
        ("-9/4", Fraction(-9, 4)),
        ("18/8", Fraction(9, 4)),
        ("0009/0004", Fraction(9, 4)),
        ("0/5", Fraction(0)),
        ("-0/5", Fraction(0)),
        ("  9/4  ", Fraction(9, 4)),
    ],
)
def test_slash_fraction_is_exact(content, expected):
    assert parse_boxed_numeric(content) == expected
    assert extract_final_boxed_numeric(r"Reasoning. \boxed{" + content + "}") == expected


@pytest.mark.parametrize(
    "content",
    [
        "9/0", "9/00", "9/-4", "+9/4", "9/+4", "9/4/2", "9 / 4",
        "9/ 4", "9 /4", "9/4 apples", "9/4=2.25", "9//4", "1e2/3",
        "9.0/4", "/4", "9/", r"\sqrt{9/4}", "9/4%",
    ],
)
def test_slash_fraction_refuses_ambiguous_or_invalid_forms(content):
    assert parse_boxed_numeric(content) is None


def test_cohort_gold_accepts_slash_but_preserves_single_box_rule():
    gold = classify_boxed_numeric_solution(r"Steps. \boxed{9/4}")
    assert gold.eligible
    assert gold.gt_boxed == "9/4"
    assert gold.numerator == 9
    assert gold.denominator == 4
    assert classify_boxed_numeric_solution(r"\boxed{9/0}").exclusion_reason == "zero_denominator"
    assert classify_boxed_numeric_solution(r"\boxed{1} \boxed{9/4}").exclusion_reason == "multiple_boxed"
    assert not classify_boxed_numeric_solution(r"\boxed{9 / 4}").eligible


def test_slash_equals_decimal_and_latex_for_binary_reward():
    for answer in ("9/4", "18/8", "2.25", r"\frac{9}{4}"):
        completion = r"Final: \boxed{" + answer + "}"
        assert verify_boxed_numeric(completion, numerator=9, denominator=4)


def test_v2_manifest_rejects_stale_v1_preparation(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"cohort": {"name": "boxed-numeric-v1"}}))
    with pytest.raises(ValueError, match="data prepare"):
        assert_current_cohort(tmp_path)
    manifest.write_text(json.dumps({"cohort": {"name": COHORT_NAME}}))
    assert_current_cohort(tmp_path)
    manifest.write_text("{}")
    with pytest.raises(ValueError, match="Invalid processed cohort manifest"):
        assert_current_cohort(tmp_path)
