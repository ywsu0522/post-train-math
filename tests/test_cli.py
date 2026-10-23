import json
from pathlib import Path

import pandas as pd

from post_train_math.cli import build_parser, run_eval


def test_eval_scores_continuations_and_preserves_native_bos(tmp_path, monkeypatch):
    import torch
    import transformers

    from post_train_math import data, models

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(
        data,
        "validate_processed_manifest",
        lambda _: (
            {"cohort": {"name": "boxed-numeric"}},
            "manifest-hash",
        ),
    )
    monkeypatch.setattr(
        data,
        "load_cohort",
        lambda _, split: pd.DataFrame(
            [
                {"problem": "one", "gt_numerator": 1, "gt_denominator": 2},
                {"problem": "two", "gt_numerator": 1, "gt_denominator": 2},
            ]
        ),
    )
    monkeypatch.setattr(models, "validate_model_source", lambda *_: {"pinned": True})

    class Encoded(dict):
        def to(self, device):
            assert device == "cuda"
            return self

    class Tokenizer:
        chat_template = "native with BOS"
        pad_token_id = 0
        eos_token_id = 9

        def apply_chat_template(self, messages, **kwargs):
            assert kwargs == {"tokenize": False, "add_generation_prompt": True}
            return "BOS " + messages[-1]["content"]

        def __call__(self, prompts, **kwargs):
            assert len(prompts) == 2 and all(p.startswith("BOS ") for p in prompts)
            assert kwargs["add_special_tokens"] is False
            assert kwargs["max_length"] == 2048 and kwargs["truncation"]
            return Encoded(input_ids=torch.tensor([[1, 2], [1, 3]]))

        def batch_decode(self, ids, **kwargs):
            assert ids.tolist() == [[4, 9], [5, 9]]
            return [r"\boxed{50\%}", r"\boxed{1/2} \boxed{"]

    class Model:
        def to(self, device):
            assert device == "cuda"
            return self

        def eval(self):
            pass

        def generate(self, **kwargs):
            assert kwargs["do_sample"] is False
            assert kwargs["max_new_tokens"] == 2048
            return torch.tensor([[1, 2, 4, 9], [1, 3, 5, 9]])

    monkeypatch.setattr(models, "load_tokenizer", lambda _: Tokenizer())
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        lambda *_args, **_kwargs: Model(),
    )
    output = tmp_path / "eval"
    args = build_parser().parse_args(
        [
            "eval",
            "--model-key",
            "olmo2",
            "--output-dir",
            str(output),
            "--batch-size",
            "2",
        ]
    )
    run_eval(args)
    report = json.loads((output / "eval_report.json").read_text())
    assert report["total"] == 2 and report["correct"] == 1 and report["accuracy"] == 0.5
    predictions = [
        json.loads(line)
        for line in (output / "predictions.jsonl").read_text().splitlines()
    ]
    assert [row["correct"] for row in predictions] == [True, False]
    assert args.data_dir == Path("data/processed")
