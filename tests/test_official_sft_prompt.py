"""Frozen boxed 1-shot prompt and portable official SFT base contract."""

import json

import pytest

from posttrain_math.artifacts import resolve_local_base_model
from posttrain_math.cli import build_parser
from posttrain_math.prompting import get_prompt_formatter
from posttrain_math.resources import (
    OFFICIAL_SFT_MODEL_DIR,
    OFFICIAL_SFT_MODEL_REPO,
    OFFICIAL_SFT_MODEL_REVISION,
)


def test_fixed_one_shot_prompt_is_registered_and_identical_across_commands():
    f = get_prompt_formatter("boxed-1shot")
    prompt = f("  Compute 6 + 7?  ")
    assert prompt.startswith(
        "Follow the example below. Solve the problem step by step.\n"
    )
    assert prompt.count(r"\boxed{1}") == 1
    assert r"\boxed{...}" in prompt
    assert "What is 3/4 + 1/4?" in prompt
    assert prompt.endswith(
        "Problem:\nCompute 6 + 7?\n\nSolution:\n"
    )
    assert f("Other problem").count("What is 3/4 + 1/4?") == 1
    for argv in (
        ["train", "sft", "--prompt", "boxed-1shot", "--levels", "1", "2", "3"],
        ["eval", "--prompt", "boxed-1shot", "--levels", "1", "2", "3"],
    ):
        assert build_parser().parse_args(argv).prompt == "boxed-1shot"


def test_official_sft_pin_is_immutable():
    assert OFFICIAL_SFT_MODEL_REPO == "allenai/OLMo-2-0425-1B-SFT"
    assert len(OFFICIAL_SFT_MODEL_REVISION) == 40
    assert OFFICIAL_SFT_MODEL_DIR.name == "olmo-2-0425-1b-sft"


def test_official_sft_adapter_base_resolution_across_runtime(tmp_path, monkeypatch):
    """An adapter created in Kaggle is resolvable after moving to Colab."""
    from posttrain_math import artifacts

    base = tmp_path / "official-sft"
    base.mkdir()
    (base / "source.json").write_text(json.dumps({
        "repo_id": OFFICIAL_SFT_MODEL_REPO,
        "resolved_commit": OFFICIAL_SFT_MODEL_REVISION,
    }))
    adapter = tmp_path / "checkpoint"
    adapter.mkdir()
    (adapter / "base_model_source.json").write_text(json.dumps({
        "repo_id": OFFICIAL_SFT_MODEL_REPO,
        "resolved_commit": OFFICIAL_SFT_MODEL_REVISION,
        "training_local_dir": "/kaggle/working/post-train-math/models/olmo-2-0425-1b-sft",
    }))
    monkeypatch.setattr(artifacts, "OFFICIAL_SFT_MODEL_DIR", base)
    assert resolve_local_base_model(adapter, OFFICIAL_SFT_MODEL_REPO) == base

    (base / "source.json").write_text(json.dumps({
        "repo_id": OFFICIAL_SFT_MODEL_REPO,
        "resolved_commit": "a" * 40,
    }))
    with pytest.raises(FileNotFoundError, match="model download"):
        resolve_local_base_model(adapter, OFFICIAL_SFT_MODEL_REPO)


def test_existing_model_directory_refuses_wrong_source(tmp_path):
    from posttrain_math.resources import download_model

    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (model / "source.json").write_text(json.dumps({
        "repo_id": OFFICIAL_SFT_MODEL_REPO,
        "requested_revision": OFFICIAL_SFT_MODEL_REVISION,
        "resolved_commit": OFFICIAL_SFT_MODEL_REVISION,
    }))
    with pytest.raises(ValueError, match="does not match"):
        download_model(
            repo_id="allenai/OLMo-2-0425-1B",
            revision="13cece9360d59bb9db636273ea8d000b67fcc27b",
            output_dir=model,
        )
    assert download_model(
        repo_id=OFFICIAL_SFT_MODEL_REPO,
        revision=OFFICIAL_SFT_MODEL_REVISION,
        output_dir=model,
    ) == model
