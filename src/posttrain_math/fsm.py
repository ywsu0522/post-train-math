"""Syntax-only, byte-level boxed-numeric masking; independent of the verifier."""
from __future__ import annotations

import json

import torch
from transformers import LogitsProcessor

MARKER = b"\\boxed{"
INVALID = -1
# States 0..5 search for the marker. State 6 means literal ``\boxed`` has
# already been generated, so the next byte is committed to ``{``.
START, SIGN, INTEGER, DOT, DECIMAL, PERCENT_SLASH, PERCENT = range(7, 14)
BOXED = START - 1
FRAC_F, FRAC_R, FRAC_A, FRAC_C, FRAC_OPEN = range(14, 19)
NUM_START, NUM_DIGITS, DEN_OPEN, DEN_START, DEN_DIGITS, FRAC_END = range(19, 25)
ACCEPTING = {INTEGER, DECIMAL, PERCENT, FRAC_END}


def advance(state: int, value: int) -> int:
    """Consume one byte, distinguishing fraction braces from the outer box."""
    if state == INVALID:
        return INVALID
    if state < START:
        if value == MARKER[state]:
            return state + 1
        if state == BOXED:
            return INVALID
        return 1 if value == MARKER[0] else 0
    digit = ord("0") <= value <= ord("9")
    if state in (START, SIGN):
        if digit:
            return INTEGER
        if value == ord("\\"):
            return FRAC_F
        if state == START and value == ord("-"):
            return SIGN
    elif state in (INTEGER, DECIMAL):
        if digit:
            return state
        if state == INTEGER and value == ord("."):
            return DOT
        if value == ord("\\"):
            return PERCENT_SLASH
    elif state == DOT and digit:
        return DECIMAL
    elif state == PERCENT_SLASH and value == ord("%"):
        return PERCENT
    elif FRAC_F <= state <= FRAC_OPEN:
        if value == b"frac{"[state - FRAC_F]:
            return state + 1
    elif state in (NUM_START, NUM_DIGITS):
        if digit:
            return NUM_DIGITS
        if state == NUM_DIGITS and value == ord("}"):
            return DEN_OPEN
    elif state == DEN_OPEN and value == ord("{"):
        return DEN_START
    elif state in (DEN_START, DEN_DIGITS):
        if digit:
            return DEN_DIGITS
        if state == DEN_DIGITS and value == ord("}"):
            return FRAC_END
    if state in ACCEPTING and value == ord("}"):
        return 0
    return INVALID


def consume(state: int, data: bytes) -> int:
    for value in data:
        state = advance(state, value)
        if state == INVALID:
            break
    return state


def final_box_matches_grammar(text: str) -> bool:
    """Measure final-box syntax independently of canonicalization and reward."""
    start = text.rfind(r"\boxed")
    if start < 0 or not text.startswith(r"\boxed{", start):
        return False
    state = START
    for value in text[start + len(MARKER):].encode("utf-8"):
        state = advance(state, value)
        if state == INVALID:
            return False
        if state == 0:
            return True
    return False


def _token_bytes(tokenizer) -> list[bytes | None]:
    """Read lossless ByteLevel token bytes, never decode partial UTF-8 alone."""
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        raise ValueError("Boxed FSM requires a fast tokenizer with a ByteLevel decoder.")
    config = json.loads(backend.to_str())
    if (config.get("decoder") or {}).get("type") != "ByteLevel":
        raise ValueError("Boxed FSM supports ByteLevel decoders (including pinned OLMo) only.")
    byte_values = [*range(33, 127), *range(161, 173), *range(174, 256)]
    extra = [value for value in range(256) if value not in byte_values]
    inverse = {chr(value): value for value in byte_values}
    inverse.update({chr(256 + index): value for index, value in enumerate(extra)})
    added = {item["id"]: item["content"] for item in config.get("added_tokens", [])}
    special = set(tokenizer.all_special_ids)
    vocab = tokenizer.get_vocab()
    pieces: list[bytes | None] = [None] * (max(vocab.values()) + 1)
    for token, token_id in vocab.items():
        if token_id in special:
            continue
        if token_id in added:
            pieces[token_id] = added[token_id].encode("utf-8")
        else:
            pieces[token_id] = bytes(inverse[char] for char in token)
    return pieces


class BoxedNumericLogitsProcessor(LogitsProcessor):
    """Mask tokens whose entire byte sequence takes the FSM into failure.

    Detect literal ``\\boxed`` in generated tokens only and then force ``{``
    before applying the answer grammar. Outside boxes special tokens retain
    their normal behavior; inside, all special tokens (including EOS) are masked.
    No box is forced, repaired, or used to terminate the completion.
    """

    def __init__(self, tokenizer, *, prompt_length: int) -> None:
        self.prompt_length = prompt_length
        self.pieces = _token_bytes(tokenizer)
        self.special_ids = set(tokenizer.all_special_ids)
        self.transitions: dict[int, list[int]] = {}
        self.masks: dict[tuple[int, int, torch.device], torch.Tensor] = {}
        self.histories: list[tuple[list[int], int]] = []

    def reset(self, *, prompt_length: int) -> None:
        """Start a fresh batch, retaining tokenizer/state mask caches."""
        self.prompt_length = prompt_length
        self.histories = []

    def _transitions(self, state: int) -> list[int]:
        if state not in self.transitions:
            self.transitions[state] = [
                state if token_id in self.special_ids and state < START else
                consume(state, piece) if piece else INVALID
                for token_id, piece in enumerate(self.pieces)
            ]
        return self.transitions[state]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        histories = []
        for row, tokens in enumerate(input_ids[:, self.prompt_length:].tolist()):
            previous, state = self.histories[row] if row < len(self.histories) else ([], 0)
            if tokens[:len(previous)] != previous:
                previous, state = [], 0
            for token_id in tokens[len(previous):]:
                state = self._transitions(state)[token_id]
                if state == INVALID:
                    raise RuntimeError("Generated stream violated boxed FSM masking.")
            histories.append((tokens, state))
            key = (state, scores.shape[-1], scores.device)
            if key not in self.masks:
                transitions = self._transitions(state)
                mask = torch.ones(scores.shape[-1], dtype=torch.bool, device=scores.device)
                count = min(len(transitions), len(mask))
                mask[:count] = torch.tensor(
                    [target == INVALID for target in transitions[:count]],
                    dtype=torch.bool, device=scores.device,
                )
                self.masks[key] = mask
            scores[row].masked_fill_(self.masks[key], -float("inf"))
            if not torch.isfinite(scores[row]).any():
                raise RuntimeError("No finite legal token remains under boxed FSM masking.")
        self.histories = histories
        return scores
