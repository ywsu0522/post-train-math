import json
from pathlib import Path

import pandas as pd

from post_train_math.cli import build_parser, run_eval, run_eval_series


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

    def test_cohort(_, split):
        assert split == "test"
        return pd.DataFrame(
            [
                {"problem": "one", "gt_numerator": 1, "gt_denominator": 2},
                {"problem": "two", "gt_numerator": 1, "gt_denominator": 2},
            ]
        )

    monkeypatch.setattr(data, "load_cohort", test_cohort)
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
            "--test-set",
            "full",
        ]
    )
    run_eval(args)
    report = json.loads((output / "eval_report.json").read_text())
    assert report["total"] == 2 and report["correct"] == 1 and report["accuracy"] == 0.5
    assert report["split"] == "test"
    assert report["boxed_output_rate"] == 0.5
    assert report["numeric_boxed_output_rate"] == 0.5
    assert report["rollout_iterations"] == 0
    predictions = [
        json.loads(line)
        for line in (output / "predictions.jsonl").read_text().splitlines()
    ]
    assert [row["correct"] for row in predictions] == [True, False]
    assert args.data_dir == Path("data/processed")


def test_series_uses_the_same_test_set_at_all_planned_steps(tmp_path, monkeypatch):
    from post_train_math import cli, curves, data, evaluation

    monkeypatch.setattr(
        data, "validate_processed_manifest", lambda _: ({}, "data-hash")
    )
    checked, evaluated = [], []

    def checkpoint(path, spec, digest):
        assert digest == "data-hash" and spec.key == "olmo2"
        step = int(path.name.removeprefix("checkpoint-"))
        checked.append(step)
        return {"rollout_iterations": step}

    monkeypatch.setattr(evaluation, "adapter_metadata", checkpoint)

    def evaluate(point):
        assert point.test_set == "fixed512" and point.limit is None
        assert point.series_name == "RLOO"
        step = (
            int(point.adapter_dir.name.removeprefix("checkpoint-"))
            if point.adapter_dir
            else 0
        )
        evaluated.append(step)
        point.output_dir.mkdir(parents=True)
        (point.output_dir / "eval_report.json").write_text(json.dumps({"step": step}))

    monkeypatch.setattr(cli, "run_eval", evaluate)

    def plot(paths, output):
        assert output == tmp_path / "evaluated/plots"
        assert [json.loads(path.read_text())["step"] for path in paths] == [
            0,
            50,
            100,
            150,
            200,
        ]

    monkeypatch.setattr(curves, "plot_reports", plot)
    args = build_parser().parse_args(
        [
            "eval-series",
            "--model-key",
            "olmo2",
            "--run-dir",
            str(tmp_path / "trained"),
            "--output-dir",
            str(tmp_path / "evaluated"),
            "--series-name",
            "RLOO",
        ]
    )
    run_eval_series(args)
    assert checked == [50, 100, 150, 200]
    assert evaluated == [0, 50, 100, 150, 200]
