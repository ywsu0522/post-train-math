import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    LogitsProcessor,
    LogitsProcessorList,
    PreTrainedTokenizerFast,
)

from posttrain_math.answers import extract_final_boxed_numeric
from posttrain_math.fsm import (
    INVALID,
    START,
    BoxedNumericLogitsProcessor,
    consume,
    final_box_matches_grammar,
)


@pytest.fixture
def tokenizer():
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    backend = Tokenizer(models.BPE(vocab={c: i for i, c in enumerate(alphabet)}, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    backend.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, eos_token="<eos>", pad_token="<pad>",
    )
    tokenizer.padding_side = "left"
    tokenizer.add_tokens([
        r"\boxed{x}", r"\boxed{2}", r"ed{x}", r"ed{2}",
        r"1} prose \boxed{", r"1} prose \boxed{x}", r"1} prose \boxed{2}",
        r"\frac{1}{0}}", "12.5", r"25\%}",
    ])
    return tokenizer


@pytest.mark.parametrize("answer", [
    "0", "-0", "001", "-12", "0.5", "-12.500", r"25\%", r"-0.5\%",
    r"\frac{12}{34}", r"-\frac{0}{00}",
])
def test_fsm_accepts_grammar_including_zero_denominator(answer):
    text = rf"work \boxed{{{answer}}} more work"
    assert consume(0, text.encode()) == 0
    assert final_box_matches_grammar(text)


@pytest.mark.parametrize("answer", [
    "", "+1", "1/2", "1e2", ".5", "5.", "1 2", " 1", "1 ", "25%",
    r"\frac{-1}{2}", r"\frac{1}{-2}", r"\frac{1} {2}", r"\frac{1}{2}\%",
    r"\sqrt{2}", "１２", "−1", "1\n2",
])
def test_fsm_rejects_invalid_grammar(answer):
    text = rf"\boxed{{{answer}}}"
    assert consume(0, text.encode()) == INVALID
    assert not final_box_matches_grammar(text)


def test_fsm_is_independent_of_verifier_and_allows_multiple_boxes():
    text = r"\boxed{1} then \boxed{\frac{1}{0}}"
    assert consume(0, text.encode()) == 0
    assert final_box_matches_grammar(text)
    assert extract_final_boxed_numeric(text) is None
    assert extract_final_boxed_numeric(r"\boxed{1} then \boxed{2}") == 2
    assert not final_box_matches_grammar(r"\boxed{1} then \boxed")


def masked(tokenizer, prefix, *, prompt=""):
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    tokens = tokenizer.encode(prefix, add_special_tokens=False)
    processor = BoxedNumericLogitsProcessor(tokenizer, prompt_length=len(prompt_ids))
    logits = processor(torch.tensor([prompt_ids + tokens], dtype=torch.long),
                       torch.zeros(1, len(tokenizer)))
    return logits[0]


def test_prospective_masking_across_marker_and_token_boundaries(tokenizer):
    for prefix, good, bad in (
        ("", r"\boxed{2}", r"\boxed{x}"),
        (r"\box", r"ed{2}", r"ed{x}"),
        (r"\boxed{", r"1} prose \boxed{2}", r"1} prose \boxed{x}"),
    ):
        logits = masked(tokenizer, prefix)
        assert torch.isfinite(logits[tokenizer.convert_tokens_to_ids(good)])
        assert torch.isneginf(logits[tokenizer.convert_tokens_to_ids(bad)])


def test_exact_boxed_marker_commits_to_open_brace(tokenizer):
    assert consume(0, b"\\boxed ") == INVALID
    logits = masked(tokenizer, r"\boxed")
    assert torch.isfinite(logits[tokenizer.convert_tokens_to_ids("{")])
    assert torch.isneginf(logits[tokenizer.convert_tokens_to_ids("x")])
    assert torch.isneginf(logits[tokenizer.convert_tokens_to_ids(" ")])
    assert torch.isneginf(logits[tokenizer.convert_tokens_to_ids("}")])
    assert torch.isneginf(logits[tokenizer.eos_token_id])


def test_only_generated_marker_activates_masking_and_closure_restores_freedom(tokenizer):
    for prefix in ("reasoning", r"\boxed{1} trailing "):
        logits = masked(tokenizer, prefix, prompt=r"System: output \boxed{")
        assert torch.isfinite(logits[tokenizer.eos_token_id])
        assert torch.isfinite(logits[tokenizer.convert_tokens_to_ids("x")])
    for prefix in (r"\boxed", r"\boxed{", r"\boxed{-", r"\boxed{1.", r"\boxed{\frac{1}{0}"):
        logits = masked(tokenizer, prefix)
        assert torch.isneginf(logits[tokenizer.eos_token_id])
        assert torch.isneginf(logits[tokenizer.convert_tokens_to_ids("x")])
    for prefix in (r"\boxed", r"\boxed{", r"\boxed{-", r"\boxed{1."):
        assert torch.isneginf(masked(tokenizer, prefix)[tokenizer.convert_tokens_to_ids("}")])
    assert torch.isfinite(masked(tokenizer, r"\boxed{1")[tokenizer.convert_tokens_to_ids("}")])


def test_left_padded_batches_independent_and_reset(tokenizer):
    prompts = tokenizer([r"long system \boxed{", "short"], padding=True, return_tensors="pt")
    width = prompts.input_ids.shape[1]
    prefixes = [r"\boxed{", "abcdefg"]
    tails = torch.tensor([tokenizer.encode(text) for text in prefixes])
    processor = BoxedNumericLogitsProcessor(tokenizer, prompt_length=width)
    logits = processor(torch.cat([prompts.input_ids, tails], dim=1), torch.zeros(2, len(tokenizer)))
    assert torch.isneginf(logits[0, tokenizer.eos_token_id])
    assert torch.isfinite(logits[1, tokenizer.eos_token_id])
    # Reordering rows must not carry one sequence's state into another.
    logits = processor(torch.cat([prompts.input_ids, tails.flip(0)], dim=1),
                       torch.zeros(2, len(tokenizer)))
    assert torch.isfinite(logits[0, tokenizer.eos_token_id])
    assert torch.isneginf(logits[1, tokenizer.eos_token_id])
    processor.reset(prompt_length=width)
    logits = processor(prompts.input_ids, torch.zeros(2, len(tokenizer)))
    assert torch.isfinite(logits[:, tokenizer.eos_token_id]).all()


def test_token_bytes_preserve_unicode_and_validate_all_token_suffixes(tokenizer):
    processor = BoxedNumericLogitsProcessor(tokenizer, prompt_length=0)
    text = "推理 café " + r"\boxed{-12.5\%}"
    tokens = tokenizer.encode(text)
    assert b"".join(processor.pieces[token] for token in tokens) == text.encode()
    # A token may legally close one box and continue free text or even start a
    # later box; the transition oracle must process the complete byte sequence.
    assert consume(START, b"12.5\\%}") == 0
    assert consume(START, b"1} prose \\boxed{2}}") == 0
    assert consume(START, b"x}") == INVALID


def test_real_generate_applies_masks_before_selection(tokenizer):
    desired = r"work \boxed{\frac{1}{0}} then \boxed{25\%} end"
    targets = tokenizer.encode(desired) + [tokenizer.eos_token_id]
    prompt_ids = tokenizer.encode("Prompt")
    width = len(prompt_ids)
    fsm = BoxedNumericLogitsProcessor(tokenizer, prompt_length=width)

    class AdversarialLogits(LogitsProcessor):
        def __call__(self, input_ids, scores):
            index = input_ids.shape[1] - width
            scores.fill_(-100)
            scores[:, targets[index]] = 10
            state = consume(0, b"".join(fsm.pieces[t] for t in targets[:index]))
            if state >= START:
                scores[:, tokenizer.eos_token_id] = 100
                scores[:, tokenizer.convert_tokens_to_ids("x")] = 99
            return scores

    model = GPT2LMHeadModel(GPT2Config(
        vocab_size=len(tokenizer), n_positions=128, n_embd=16, n_layer=1, n_head=1,
        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
    )).eval()
    inputs = torch.tensor([prompt_ids])
    with torch.inference_mode():
        output = model.generate(
            inputs, attention_mask=torch.ones_like(inputs), max_new_tokens=len(targets),
            do_sample=False, logits_processor=LogitsProcessorList([AdversarialLogits(), fsm]),
        )
    assert output[0, width:].tolist() == targets
    assert tokenizer.decode(output[0, width:], skip_special_tokens=True) == desired


def test_unsupported_decoder_fails_explicitly(tokenizer):
    tokenizer.backend_tokenizer.decoder = decoders.WordPiece()
    with pytest.raises(ValueError, match="ByteLevel"):
        BoxedNumericLogitsProcessor(tokenizer, prompt_length=0)
