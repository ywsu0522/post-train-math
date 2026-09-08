from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction

_INTEGER_RE = re.compile(r"^[+-]?[0-9]+$")
_SLASH_FRACTION_RE = re.compile(
    r"^([+-]?[0-9]+)\s*/\s*([+-]?[0-9]+)$"
)
_LATEX_FRACTION_RE = re.compile(
    r"^([+-]?)\s*\\frac\s*"
    r"\{\s*([+-]?[0-9]+)\s*\}\s*"
    r"\{\s*([+-]?[0-9]+)\s*\}$"
)


@dataclass(frozen=True)
class BoxedScan:
    valid_contents: tuple[str, ...]
    empty_count: int
    unbraced_count: int
    malformed_count: int
    marker_count: int


@dataclass(frozen=True)
class RationalGold:
    eligible: bool
    gt_boxed: str | None
    numerator: int | None
    denominator: int | None
    exclusion_reason: str | None

    @property
    def fraction(self) -> Fraction | None:
        if self.numerator is None or self.denominator is None:
            return None
        return Fraction(self.numerator, self.denominator)


def _is_escaped(text: str, index: int) -> bool:
    backslashes = 0
    index -= 1
    while index >= 0 and text[index] == "\\":
        backslashes += 1
        index -= 1
    return backslashes % 2 == 1


def _find_matching_brace(text: str, open_index: int) -> int | None:
    depth = 0
    for index in range(open_index, len(text)):
        char = text[index]
        if char == "{" and not _is_escaped(text, index):
            depth += 1
        elif char == "}" and not _is_escaped(text, index):
            depth -= 1
            if depth == 0:
                return index
    return None


def _boxed_content_at(text: str, marker_index: int) -> tuple[str | None, str]:
    """Parse the single ``\\boxed`` marker starting at ``marker_index``.

    Returns ``(content, status)`` where status is one of ``valid``, ``empty``,
    ``unbraced`` or ``malformed``.
    """
    cursor = marker_index + len(r"\boxed")
    while cursor < len(text) and text[cursor].isspace():
        cursor += 1

    if cursor >= len(text):
        return None, "malformed"
    if text[cursor] != "{":
        return None, "unbraced"

    close_index = _find_matching_brace(text, cursor)
    if close_index is None:
        return None, "malformed"

    content = text[cursor + 1 : close_index].strip()
    if not content:
        return None, "empty"
    return content, "valid"


def scan_boxed(text: str) -> BoxedScan:
    marker = r"\boxed"
    valid_contents: list[str] = []
    empty_count = 0
    unbraced_count = 0
    malformed_count = 0
    marker_count = 0

    index = 0
    while True:
        start = text.find(marker, index)
        if start == -1:
            break

        marker_count += 1
        content, status = _boxed_content_at(text, start)
        if status == "valid":
            assert content is not None
            valid_contents.append(content)
        elif status == "empty":
            empty_count += 1
        elif status == "unbraced":
            unbraced_count += 1
        else:
            malformed_count += 1

        index = start + len(marker)

    return BoxedScan(
        valid_contents=tuple(valid_contents),
        empty_count=empty_count,
        unbraced_count=unbraced_count,
        malformed_count=malformed_count,
        marker_count=marker_count,
    )


def classify_boxed_format(text: str) -> str:
    scan = scan_boxed(text)
    if scan.marker_count == 0:
        return "no_boxed"
    if scan.marker_count > 1:
        return "multiple_boxed"
    if scan.empty_count:
        return "empty_boxed"
    if scan.unbraced_count or scan.malformed_count:
        return "malformed_boxed"
    if len(scan.valid_contents) == 1:
        return "single_valid"
    return "malformed_boxed"


def extract_final_boxed(text: str) -> str | None:
    """Return the content of the final ``\\boxed`` marker only.

    A malformed final marker does not fall back to an earlier valid box.
    """
    marker = r"\boxed"
    start = text.rfind(marker)
    if start == -1:
        return None
    content, status = _boxed_content_at(text, start)
    return content if status == "valid" else None


def _parse_exact_rational_with_reason(
    text: str | None,
) -> tuple[Fraction | None, str | None]:
    if not isinstance(text, str):
        return None, "non_rational_boxed"

    value = text.strip()
    if not value:
        return None, "non_rational_boxed"

    if _INTEGER_RE.fullmatch(value):
        return Fraction(int(value), 1), None

    slash_match = _SLASH_FRACTION_RE.fullmatch(value)
    if slash_match is not None:
        numerator = int(slash_match.group(1))
        denominator = int(slash_match.group(2))
        if denominator == 0:
            return None, "zero_denominator"
        return Fraction(numerator, denominator), None

    latex_match = _LATEX_FRACTION_RE.fullmatch(value)
    if latex_match is not None:
        outer_sign = -1 if latex_match.group(1) == "-" else 1
        numerator = outer_sign * int(latex_match.group(2))
        denominator = int(latex_match.group(3))
        if denominator == 0:
            return None, "zero_denominator"
        return Fraction(numerator, denominator), None

    return None, "non_rational_boxed"


def parse_exact_rational(text: str | None) -> Fraction | None:
    """Parse the exact-rational-v1 answer grammar.

    Accepted forms are signed integers, ``a/b``, and LaTeX ``\\frac{a}{b}``
    with an optional sign before ``\\frac``. Decimal, symbolic, unit-bearing,
    radical, tuple, interval and prose answers are intentionally rejected.
    """
    value, _ = _parse_exact_rational_with_reason(text)
    return value


def classify_exact_rational_solution(solution: str) -> RationalGold:
    """Classify a gold MATH solution for the exact-rational-v1 cohort."""
    scan = scan_boxed(solution)

    if scan.marker_count == 0:
        return RationalGold(False, None, None, None, "no_boxed")
    if scan.marker_count > 1:
        return RationalGold(False, None, None, None, "multiple_boxed")
    if scan.empty_count:
        return RationalGold(False, None, None, None, "empty_boxed")
    if scan.unbraced_count or scan.malformed_count:
        return RationalGold(False, None, None, None, "malformed_boxed")
    if len(scan.valid_contents) != 1:
        return RationalGold(False, None, None, None, "malformed_boxed")

    content = scan.valid_contents[0]
    fraction, reason = _parse_exact_rational_with_reason(content)
    if fraction is None:
        return RationalGold(False, content, None, None, reason)

    return RationalGold(
        eligible=True,
        gt_boxed=content,
        numerator=fraction.numerator,
        denominator=fraction.denominator,
        exclusion_reason=None,
    )


def extract_final_boxed_rational(text: str) -> Fraction | None:
    return parse_exact_rational(extract_final_boxed(text))


def verify_exact_rational(
    completion: str,
    *,
    numerator: int,
    denominator: int,
) -> bool:
    if denominator == 0:
        raise ValueError("Gold denominator must be non-zero.")
    prediction = extract_final_boxed_rational(completion)
    return prediction == Fraction(numerator, denominator)
