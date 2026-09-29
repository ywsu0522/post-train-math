from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, DecimalException, localcontext
from fractions import Fraction

_NUMBER_RE = re.compile(r"(-?[0-9]+(?:\.[0-9]+)?)(\\%)?")
_LATEX_FRACTION_RE = re.compile(
    r"(-?)\\frac\{([0-9]+)\}\{([0-9]+)\}"
)


@dataclass(frozen=True)
class BoxedScan:
    valid_contents: tuple[str, ...]
    empty_count: int
    unbraced_count: int
    malformed_count: int
    marker_count: int


@dataclass(frozen=True)
class NumericGold:
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


def _parse_boxed_numeric_with_reason(
    text: str | None,
) -> tuple[Fraction | None, str | None]:
    if not isinstance(text, str):
        return None, "non_numeric_boxed"

    value = text.strip()
    if not value:
        return None, "non_numeric_boxed"

    try:
        number_match = _NUMBER_RE.fullmatch(value)
        if number_match is not None:
            number = Decimal(number_match.group(1))
            if number_match.group(2):
                # Decimal construction is exact; division also needs enough
                # precision to preserve every input digit, regardless of the
                # caller's current Decimal precision.
                with localcontext() as context:
                    context.prec = max(1, len(number.as_tuple().digits))
                    number /= 100
            return Fraction(number), None

        latex_match = _LATEX_FRACTION_RE.fullmatch(value)
        if latex_match is not None:
            outer_sign = -1 if latex_match.group(1) == "-" else 1
            numerator = outer_sign * int(latex_match.group(2))
            denominator = int(latex_match.group(3))
            return Fraction(numerator, denominator), None
    except ZeroDivisionError:
        return None, "zero_denominator"
    except (ValueError, DecimalException):
        return None, "non_numeric_boxed"

    return None, "non_numeric_boxed"


def parse_boxed_numeric(text: str | None) -> Fraction | None:
    """Parse the boxed-numeric-v1 answer grammar.

    Strip outer whitespace, then full-match a number (optional minus, digits,
    optional decimal digits, optional ``\\%``) or ``-?\\frac{digits}{digits}``.
    Numbers use Decimal, percentages divide by 100, and both paths produce
    canonical Fractions. Parse failures, including zero denominators, are invalid.
    """
    value, _ = _parse_boxed_numeric_with_reason(text)
    return value


def classify_boxed_numeric_solution(solution: str) -> NumericGold:
    """Classify a gold MATH solution for the boxed-numeric-v1 cohort."""
    scan = scan_boxed(solution)

    if scan.marker_count == 0:
        return NumericGold(False, None, None, None, "no_boxed")
    if scan.marker_count > 1:
        return NumericGold(False, None, None, None, "multiple_boxed")
    if scan.empty_count:
        return NumericGold(False, None, None, None, "empty_boxed")
    if scan.unbraced_count or scan.malformed_count:
        return NumericGold(False, None, None, None, "malformed_boxed")
    if len(scan.valid_contents) != 1:
        return NumericGold(False, None, None, None, "malformed_boxed")

    content = scan.valid_contents[0]
    fraction, reason = _parse_boxed_numeric_with_reason(content)
    if fraction is None:
        return NumericGold(False, content, None, None, reason)

    return NumericGold(
        eligible=True,
        gt_boxed=content,
        numerator=fraction.numerator,
        denominator=fraction.denominator,
        exclusion_reason=None,
    )


def normalize_boxed_numeric_gold_solution(solution: str) -> str:
    """Normalize the single eligible gold box to the exact SFT/FSM spelling.

    Preserve all reasoning text and the accepted boxed-content serialization,
    but rewrite optional whitespace between ``\\boxed`` and ``{`` and outer
    whitespace inside the box as exactly ``\\boxed{<gt_boxed>}``.
    """
    classification = classify_boxed_numeric_solution(solution)
    if not classification.eligible or classification.gt_boxed is None:
        raise ValueError("Gold solution is not eligible for boxed-numeric-v1.")

    marker = r"\boxed"
    start = solution.find(marker)
    if start < 0:
        raise AssertionError("Eligible boxed-numeric solution has no marker.")

    open_index = start + len(marker)
    while open_index < len(solution) and solution[open_index].isspace():
        open_index += 1
    if open_index >= len(solution) or solution[open_index] != "{":
        raise AssertionError("Eligible boxed-numeric solution has no opening brace.")

    close_index = _find_matching_brace(solution, open_index)
    if close_index is None:
        raise AssertionError("Eligible boxed-numeric solution has no closing brace.")

    normalized_box = rf"\boxed{{{classification.gt_boxed}}}"
    return solution[:start] + normalized_box + solution[close_index + 1 :]


def extract_final_boxed_numeric(text: str) -> Fraction | None:
    return parse_boxed_numeric(extract_final_boxed(text))


def verify_boxed_numeric(
    completion: str,
    *,
    numerator: int,
    denominator: int,
) -> bool:
    if denominator == 0:
        raise ValueError("Gold denominator must be non-zero.")
    prediction = extract_final_boxed_numeric(completion)
    return prediction == Fraction(numerator, denominator)
