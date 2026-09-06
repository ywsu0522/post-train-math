import json
from pathlib import Path

from posttrain_math import artifacts


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    path.write_text(
        json.dumps(value),
        encoding="utf-8",
    )


def test_normalize_adapter_records_canonical_base(
    tmp_path,
) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()

    _write_json(
        adapter / "adapter_config.json",
        {
            "base_model_name_or_path":
                "models/local",
            "revision": None,
            "peft_type": "LORA",
        },
    )

    source = {
        "repo_id": "org/model",
        "requested_revision": "main",
        "resolved_commit": "abc123",
        "training_local_dir":
            "models/local",
        "portable": True,
    }

    assert artifacts.normalize_adapter_artifact(
        adapter,
        source,
    )

    config = json.loads(
        (
            adapter
            / "adapter_config.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    assert (
        config[
            "base_model_name_or_path"
        ]
        == "org/model"
    )

    assert (
        config["revision"]
        == "abc123"
    )

    recorded = json.loads(
        (
            adapter
            / "base_model_source.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    assert recorded == source


def test_resolve_adapter_uses_matching_local_copy(
    tmp_path,
    monkeypatch,
) -> None:
    base = tmp_path / "base"
    base.mkdir()

    _write_json(
        base / "source.json",
        {
            "repo_id":
                artifacts.DEFAULT_MODEL_REPO,
            "resolved_commit":
                "abc123",
        },
    )

    adapter = tmp_path / "adapter"
    adapter.mkdir()

    _write_json(
        adapter / "adapter_config.json",
        {
            "base_model_name_or_path":
                artifacts.DEFAULT_MODEL_REPO,
        },
    )

    source = {
        "repo_id":
            artifacts.DEFAULT_MODEL_REPO,
        "requested_revision":
            "abc123",
        "resolved_commit":
            "abc123",
        "training_local_dir":
            "missing/path",
        "portable": True,
    }

    _write_json(
        adapter
        / "base_model_source.json",
        source,
    )

    monkeypatch.setattr(
        artifacts,
        "DEFAULT_MODEL_DIR",
        base,
    )

    resolved = (
        artifacts.resolve_local_base_model(
            adapter,
            artifacts.DEFAULT_MODEL_REPO,
        )
    )

    assert resolved == base


def test_staging_keeps_published_adapter_unchanged(
    tmp_path,
    monkeypatch,
) -> None:
    base = tmp_path / "base"
    base.mkdir()

    _write_json(
        base / "source.json",
        {
            "repo_id":
                artifacts.DEFAULT_MODEL_REPO,
            "resolved_commit":
                "abc123",
        },
    )

    adapter = tmp_path / "adapter"
    adapter.mkdir()

    _write_json(
        adapter / "adapter_config.json",
        {
            "base_model_name_or_path":
                artifacts.DEFAULT_MODEL_REPO,
            "revision":
                "abc123",
        },
    )

    _write_json(
        adapter
        / "base_model_source.json",
        {
            "repo_id":
                artifacts.DEFAULT_MODEL_REPO,
            "requested_revision":
                "abc123",
            "resolved_commit":
                "abc123",
            "training_local_dir":
                "missing/path",
            "portable": True,
        },
    )

    monkeypatch.setattr(
        artifacts,
        "DEFAULT_MODEL_DIR",
        base,
    )

    with (
        artifacts
        .staged_adapter_for_local_training(
            adapter
        )
    ) as (
        staged,
        _source,
        resolved_base,
    ):
        staged_config = json.loads(
            (
                staged
                / "adapter_config.json"
            ).read_text(
                encoding="utf-8"
            )
        )

        assert (
            staged_config[
                "base_model_name_or_path"
            ]
            == str(base)
        )

        assert resolved_base == base

    original_config = json.loads(
        (
            adapter
            / "adapter_config.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    assert (
        original_config[
            "base_model_name_or_path"
        ]
        == artifacts.DEFAULT_MODEL_REPO
    )
