from __future__ import annotations

from dataclasses import dataclass
import logging
import os

from math_verify import parse, verify
from math_verify.parser import LatexExtractionConfig


_IS_WINDOWS = os.name == "nt"

# Math-Verify 0.9.0 has a known Windows issue when its internal
# timeout wrapper is enabled. Disable that timeout on Windows.
_PARSE_TIMEOUT_SECONDS = None if _IS_WINDOWS else 5
_VERIFY_TIMEOUT_SECONDS = None if _IS_WINDOWS else 5


class _DisabledTimeoutWarningFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "Timeout is disabled as" not in record.getMessage()


if _IS_WINDOWS:
    timeout_filter = _DisabledTimeoutWarningFilter()

    logging.getLogger(
        "math_verify.parser"
    ).addFilter(timeout_filter)

    logging.getLogger(
        "math_verify.grader"
    ).addFilter(timeout_filter)


@dataclass(frozen=True)
class BoxedScan:
    valid_contents: tuple[str, ...]
    empty_count: int
    unbraced_count: int
    malformed_count: int
    marker_count: int


def _is_escaped(
    text: str,
    index: int,
) -> bool:
    backslashes = 0
    index -= 1

    while (
        index >= 0
        and text[index] == "\\"
    ):
        backslashes += 1
        index -= 1

    return backslashes % 2 == 1


def _find_matching_brace(
    text: str,
    open_index: int,
) -> int | None:
    depth = 0

    for index in range(
        open_index,
        len(text),
    ):
        char = text[index]

        if (
            char == "{"
            and not _is_escaped(text, index)
        ):
            depth += 1

        elif (
            char == "}"
            and not _is_escaped(text, index)
        ):
            depth -= 1

            if depth == 0:
                return index

    return None


def scan_boxed(
    text: str,
) -> BoxedScan:
    """
    Scan all occurrences of '\\boxed'.

    Accepted valid forms:
        \\boxed{ANSWER}
        \\boxed {ANSWER}

    Whitespace between '\\boxed' and '{' is allowed.
    """
    marker = "\\boxed"

    valid_contents: list[str] = []

    empty_count = 0
    unbraced_count = 0
    malformed_count = 0
    marker_count = 0

    index = 0

    while True:
        start = text.find(
            marker,
            index,
        )

        if start == -1:
            break

        marker_count += 1

        cursor = (
            start + len(marker)
        )

        # Accept whitespace:
        # \boxed {17}
        while (
            cursor < len(text)
            and text[cursor].isspace()
        ):
            cursor += 1

        if cursor >= len(text):
            malformed_count += 1
            index = cursor
            continue

        if text[cursor] == "{":
            close_index = (
                _find_matching_brace(
                    text,
                    cursor,
                )
            )

            if close_index is None:
                malformed_count += 1
                index = cursor + 1
                continue

            content = text[
                cursor + 1 :
                close_index
            ].strip()

            if content:
                valid_contents.append(
                    content
                )
            else:
                empty_count += 1

            index = close_index + 1
            continue

        # Example:
        # \boxed 17
        unbraced_count += 1

        index = cursor + 1

    return BoxedScan(
        valid_contents=tuple(
            valid_contents
        ),
        empty_count=empty_count,
        unbraced_count=unbraced_count,
        malformed_count=malformed_count,
        marker_count=marker_count,
    )


def classify_boxed_format(
    text: str,
) -> str:
    """
    Return one mutually-exclusive
    row-level audit category.
    """
    scan = scan_boxed(text)

    valid_count = len(
        scan.valid_contents
    )

    if valid_count >= 2:
        return "multiple_valid"

    if scan.empty_count > 0:
        return "empty_boxed"

    if scan.unbraced_count > 0:
        return "malformed_unbraced"

    if scan.malformed_count > 0:
        return "other_malformed"

    if valid_count == 1:
        return "single_valid"

    return "no_boxed"


def extract_last_boxed(
    text: str,
) -> str | None:
    """
    Return content of the last valid boxed answer.

    Multiple valid boxed expressions are
    resolved by taking the final valid one.
    """
    scan = scan_boxed(text)

    if not scan.valid_contents:
        return None

    return scan.valid_contents[-1]


def parse_boxed_answer(
    answer: str | None,
):
    if answer is None:
        return None

    wrapped = (
        f"\\boxed{{{answer}}}"
    )

    try:
        parsed = parse(
            wrapped,
            extraction_config=[
                LatexExtractionConfig(
                    boxed_match_priority=0,
                )
            ],
            extraction_mode="first_match",
            parsing_timeout=(
                _PARSE_TIMEOUT_SECONDS
            ),
        )
    except Exception:
        return None

    return parsed if parsed else None


def verify_boxed_answers(
    gold: str | None,
    prediction: str | None,
) -> tuple[bool, bool, bool]:
    """
    Returns:
        correct,
        gold_parseable,
        prediction_parseable
    """
    gold_parsed = (
        parse_boxed_answer(gold)
    )

    if gold_parsed is None:
        return False, False, False

    prediction_parsed = (
        parse_boxed_answer(
            prediction
        )
    )

    if prediction_parsed is None:
        return False, True, False

    try:
        correct = bool(
            verify(
                gold_parsed,
                prediction_parsed,
                timeout_seconds=(
                    _VERIFY_TIMEOUT_SECONDS
                ),
            )
        )
    except Exception:
        correct = False

    return correct, True, True