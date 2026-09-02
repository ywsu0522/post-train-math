from postrain_math_lab.answers import (
    classify_boxed_format,
    extract_last_boxed,
)


def test_extract_single_boxed() -> None:
    text = r"Answer: \boxed{42}"

    assert (
        extract_last_boxed(text)
        == "42"
    )


def test_extract_boxed_with_space() -> None:
    text = r"Answer: \boxed {17}"

    assert (
        extract_last_boxed(text)
        == "17"
    )

    assert (
        classify_boxed_format(text)
        == "single_valid"
    )


def test_extract_nested_boxed() -> None:
    text = (
        r"Answer: "
        r"\boxed{\frac{1}{2}}"
    )

    assert (
        extract_last_boxed(text)
        == r"\frac{1}{2}"
    )


def test_extract_last_valid_boxed() -> None:
    text = (
        r"First \boxed {1}, "
        r"finally \boxed{2}"
    )

    assert (
        classify_boxed_format(text)
        == "multiple_valid"
    )

    assert (
        extract_last_boxed(text)
        == "2"
    )


def test_empty_boxed() -> None:
    assert (
        classify_boxed_format(
            r"\boxed{}"
        )
        == "empty_boxed"
    )


def test_unbraced_boxed() -> None:
    assert (
        classify_boxed_format(
            r"\boxed 42"
        )
        == "malformed_unbraced"
    )


def test_no_boxed() -> None:
    assert (
        extract_last_boxed(
            "Answer is 42"
        )
        is None
    )