import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from posttrain_math.baselines import (
    MODELS,
    make_chat_prompt,
    validate_model_source,
    validate_processed_manifest,
)
from posttrain_math.cli import build_parser


def test_preserved_baselines_are_original_measurements():
    root = Path(__file__).resolve().parents[1] / "baselines"
    manifest = json.loads((root / "manifest.json").read_text())
    for entry in manifest["runs"]:
        path = root / entry["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
        with zipfile.ZipFile(path) as archive:
            report = json.loads(archive.read("baseline_report.json"))
            metrics = json.loads(archive.read("metrics.json"))
            rows = [json.loads(line) for line in archive.read("predictions.jsonl").splitlines()]
        spec = MODELS[report["model_key"]]
        assert report["model_revision"] == spec.revision
        assert report["system_prompt"] == spec.system_prompt
        assert report["cohort"] == "exact-rational-v1"
        assert report["generation"]["max_new_tokens"] == 1024
        assert len(rows) == entry["num_examples"] == 544
        assert sum(row["correct"] for row in rows) == entry["num_correct"] == metrics["num_correct"]
        assert metrics["accuracy"] == entry["accuracy"]


def test_native_chat_template_uses_fixed_system_message():
    class Tokenizer:
        chat_template = "native"

        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            assert tokenize is False and add_generation_prompt is True
            assert messages == [
                {"role": "system", "content": MODELS["granite"].system_prompt},
                {"role": "user", "content": "Problem:\n1+1?"},
            ]
            return "native formatted prompt"

    assert make_chat_prompt(Tokenizer(), MODELS["granite"], " 1+1? ") == "native formatted prompt"


def test_model_revision_mismatch_is_rejected(tmp_path):
    spec = MODELS["granite"]
    (tmp_path / "source.json").write_text(json.dumps({
        "repo_id": spec.repo_id, "requested_revision": spec.revision, "resolved_commit": "wrong"
    }))
    with pytest.raises(ValueError, match="pinned"):
        validate_model_source(tmp_path, spec)


def test_current_cohort_requires_verified_source_and_unchanged_files(tmp_path):
    from posttrain_math import data

    with pytest.raises(ValueError, match="Missing processed"):
        validate_processed_manifest(tmp_path)
    paths = {split: tmp_path / f"{split}.parquet" for split in ("train", "dev", "test")}
    for split, path in paths.items():
        path.write_bytes(split.encode())
    manifest = {
        "cohort": {"name": "boxed-numeric-v2"},
        "split": {"seed": 42, "dev_ratio": 0.1, "stratify": ["type", "level"]},
        "source": {
            "repo_id": data.MATH_DATASET_REPO, "revision": data.MATH_DATASET_REVISION,
            "sha256": {"train": data.MATH_TRAIN_SHA256, "test": data.MATH_TEST_SHA256},
        },
        "processed_sha256": {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in paths.items()},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    assert validate_processed_manifest(tmp_path)[0] == manifest
    paths["dev"].write_bytes(b"changed")
    with pytest.raises(ValueError, match="dev SHA256 mismatch"):
        validate_processed_manifest(tmp_path)


def test_prepare_rejects_unverified_raw_files_before_writing(tmp_path):
    from posttrain_math.data import prepare_datasets

    train, test, output = tmp_path / "train", tmp_path / "test", tmp_path / "processed"
    train.write_bytes(b"unverified train")
    test.write_bytes(b"unverified test")
    with pytest.raises(ValueError, match="Raw MATH SHA256"):
        prepare_datasets(train, test, output, seed=42, dev_ratio=0.1)
    assert not output.exists()


def test_cli_exposes_only_data_model_and_eval():
    parser = build_parser()
    for command in ("train", "rloo", "grpo", "doctor"):
        with pytest.raises(SystemExit):
            parser.parse_args([command])
    args = parser.parse_args(["eval", "--model-key", "olmo2", "--output-dir", "runs/eval"])
    assert args.max_new_tokens == 1024


def test_generation_does_not_add_a_second_bos():
    import torch

    from posttrain_math.modeling import HFModelRunner

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 9

        def __call__(self, prompts, *, add_special_tokens, return_tensors, padding, truncation):
            assert prompts == ["native chat with BOS"]
            assert add_special_tokens is False
            return {"input_ids": torch.tensor([[1, 2]])}

        def batch_decode(self, ids, *, skip_special_tokens):
            assert ids.tolist() == [[3, 9]]
            return [r"\boxed{1}"]

    class Model:
        def generate(self, **kwargs):
            assert kwargs["input_ids"].tolist() == [[1, 2]]
            assert kwargs["do_sample"] is False
            return torch.tensor([[1, 2, 3, 9]])

    runner = HFModelRunner(
        model_path=Path("fake"), model=Model(), tokenizer=Tokenizer(),
        device=torch.device("cpu"), dtype=torch.float32, batch_size=1, rank=0, world_size=1,
    )
    assert runner.generate(["native chat with BOS"], max_new_tokens=2, do_sample=False) == [r"\boxed{1}"]
