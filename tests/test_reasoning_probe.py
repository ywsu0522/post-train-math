import pytest

from posttrain_math.reasoning_probe import (
    pre_box_token_limit,
    prefix_token_counts,
    wilson_interval,
)


def test_prefix_token_counts_are_fixed_and_unique() -> None:
    assert prefix_token_counts(100) == [
        (0.0, 0),
        (0.25, 25),
        (0.5, 50),
        (0.75, 75),
    ]
    assert prefix_token_counts(2) == [(0.0, 0), (0.25, 1)]


def test_wilson_interval_contains_empirical_rate() -> None:
    low, high = wilson_interval(6, 8)
    assert 0.0 <= low < 0.75 < high <= 1.0


def test_wilson_interval_validates_counts() -> None:
    with pytest.raises(ValueError):
        wilson_interval(2, 1)


class _FakeTokenizer:
    def __init__(self) -> None:
        self.pieces = {1: "work ", 2: r"\boxed", 3: "{2}"}

    def decode(self, token_ids, skip_special_tokens=True):
        del skip_special_tokens
        return "".join(self.pieces[token] for token in token_ids)


class _SplitMarkerTokenizer:
    def __init__(self) -> None:
        self.pieces = {1: "work ", 2: r"\bo", 3: "xed", 4: "{2}"}

    def decode(self, token_ids, skip_special_tokens=True):
        del skip_special_tokens
        return "".join(self.pieces[token] for token in token_ids)


def test_prefix_probe_excludes_explicit_boxed_answer_tokens() -> None:
    assert pre_box_token_limit(_FakeTokenizer(), [1, 2, 3]) == 1
    assert pre_box_token_limit(_FakeTokenizer(), [1]) == 1


def test_prefix_probe_excludes_partial_split_boxed_marker() -> None:
    assert pre_box_token_limit(_SplitMarkerTokenizer(), [1, 2, 3, 4]) == 1
