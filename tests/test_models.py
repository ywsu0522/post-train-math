import hashlib
import json
from pathlib import Path

import pytest

from post_train_math import models as resources
from post_train_math.cli import build_parser
from post_train_math.data import validate_processed_manifest
from post_train_math.models import (
    MODELS,
    make_chat_prompt,
    validate_model_source,
)


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

    assert (
        make_chat_prompt(Tokenizer(), MODELS["granite"], " 1+1? ")
        == "native formatted prompt"
    )


def test_model_revision_mismatch_is_rejected(tmp_path):
    spec = MODELS["granite"]
    (tmp_path / "source.json").write_text(
        json.dumps(
            {
                "repo_id": spec.repo_id,
                "requested_revision": spec.revision,
                "resolved_commit": "wrong",
            }
        )
    )
    with pytest.raises(ValueError, match="pinned"):
        validate_model_source(tmp_path, spec)


def test_current_cohort_requires_verified_source_and_unchanged_files(
    tmp_path, monkeypatch
):
    from post_train_math import data

    with pytest.raises(ValueError, match="Missing processed"):
        validate_processed_manifest(tmp_path)
    paths = {split: tmp_path / f"{split}.parquet" for split in ("train", "dev", "test")}
    for split, path in paths.items():
        path.write_bytes(split.encode())
    manifest = {
        "cohort": {"name": "boxed-numeric", "definition": data.COHORT_DEFINITION},
        "pipeline": data.PIPELINE,
        "split": {"seed": 42, "dev_ratio": 0.1, "stratify": ["type", "level"]},
        "source": {
            "repo_id": data.MATH_DATASET_REPO,
            "revision": data.MATH_DATASET_REVISION,
            "sha256": {"train": data.MATH_TRAIN_SHA256, "test": data.MATH_TEST_SHA256},
        },
        "processed_sha256": {
            key: hashlib.sha256(path.read_bytes()).hexdigest()
            for key, path in paths.items()
        },
    }
    monkeypatch.setattr(data, "FROZEN_PROCESSED_SHA256", manifest["processed_sha256"])
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    assert validate_processed_manifest(tmp_path)[0] == manifest
    paths["dev"].write_bytes(b"changed")
    with pytest.raises(ValueError, match="dev SHA256 mismatch"):
        validate_processed_manifest(tmp_path)


def test_prepare_rejects_unverified_raw_files_before_writing(tmp_path):
    from post_train_math.data import prepare_datasets

    train, test, output = tmp_path / "train", tmp_path / "test", tmp_path / "processed"
    train.write_bytes(b"unverified train")
    test.write_bytes(b"unverified test")
    with pytest.raises(ValueError, match="Raw MATH SHA256"):
        prepare_datasets(train, test, output, seed=42, dev_ratio=0.1)
    assert not output.exists()


def test_cli_exposes_four_workflows():
    parser = build_parser()
    assert parser.parse_args(["data", "prepare"]).command == "data"
    assert (
        parser.parse_args(["model", "download", "--model-key", "olmo2"]).command
        == "model"
    )
    for command in ("eval", "train"):
        args = parser.parse_args(
            [command, "--model-key", "olmo2", "--output-dir", "runs/test"]
        )
        assert args.command == command
    train = parser.parse_args(
        [
            "train",
            "--model-key",
            "olmo2",
            "--output-dir",
            "runs/test",
            "--gradient-accumulation-steps",
            "16",
            "--prepare-only",
        ]
    )
    assert train.gradient_accumulation_steps == 16
    assert train.prepare_only
    for command in ("rloo", "grpo", "doctor"):
        with pytest.raises(SystemExit):
            parser.parse_args([command])


def test_download_model_records_resolved_commit(tmp_path, monkeypatch) -> None:
    class FakeInfo:
        sha = "abc123"

    class FakeApi:
        def model_info(self, repo_id: str, revision: str):
            assert repo_id == "org/model"
            assert revision == "abc123"
            return FakeInfo()

    def fake_snapshot_download(*, repo_id, revision, local_dir):
        assert repo_id == "org/model"
        assert revision == "abc123"
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        (Path(local_dir) / "config.json").write_text("{}", encoding="utf-8")
        return str(local_dir)

    monkeypatch.setattr(resources, "HfApi", FakeApi)
    monkeypatch.setattr(resources, "snapshot_download", fake_snapshot_download)

    target = tmp_path / "models" / "model"
    resources.download_model(
        repo_id="org/model",
        output_dir=target,
        revision="abc123",
    )

    manifest = json.loads((target / "source.json").read_text(encoding="utf-8"))
    assert manifest["resolved_commit"] == "abc123"
    assert manifest["local_dir"] == str(target)


def test_download_model_reuses_complete_local_copy(tmp_path, monkeypatch) -> None:
    target = tmp_path / "models" / "model"
    target.mkdir(parents=True)
    (target / "config.json").write_text("{}", encoding="utf-8")
    (target / "source.json").write_text(
        json.dumps(
            {
                "repo_id": "org/model",
                "requested_revision": "abc123",
                "resolved_commit": "abc123",
            }
        ),
        encoding="utf-8",
    )

    class FailApi:
        def __init__(self, *args, **kwargs):
            raise AssertionError("network lookup should not run")

    monkeypatch.setattr(resources, "HfApi", FailApi)

    result = resources.download_model(
        repo_id="org/model",
        output_dir=target,
        revision="abc123",
    )

    assert result == target
