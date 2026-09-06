from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from posttrain_math.environment import environment_manifest, inspect_environment
from posttrain_math.resources import (
    DEFAULT_MODEL_DIR,
    DEFAULT_MODEL_REPO,
    DEFAULT_MODEL_REVISION,
)

BASE_MODEL_SOURCE_FILENAME = "base_model_source.json"


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )


def is_main_process() -> bool:
    try:
        return int(os.environ.get("RANK", "0")) == 0
    except ValueError:
        return True


def local_model_source(model_path: Path) -> dict[str, Any]:
    model_path = Path(model_path)
    source = _read_json(model_path / "source.json") or {}

    return {
        "repo_id": source.get("repo_id"),
        "requested_revision": source.get("requested_revision"),
        "resolved_commit": source.get("resolved_commit"),
        "training_local_dir": str(model_path),
        "portable": bool(
            source.get("repo_id")
            and source.get("resolved_commit")
        ),
    }


def adapter_base_model_source(
    adapter_path: Path,
) -> dict[str, Any] | None:
    return _read_json(
        Path(adapter_path)
        / BASE_MODEL_SOURCE_FILENAME
    )


def _source_matches(
    candidate: Path,
    source: dict[str, Any],
) -> bool:
    expected_commit = source.get(
        "resolved_commit"
    )
    expected_repo = source.get(
        "repo_id"
    )

    if not expected_commit and not expected_repo:
        return True

    local_source = _read_json(
        candidate / "source.json"
    )

    if local_source is None:
        return False

    commit_mismatch = (
        expected_commit
        and local_source.get(
            "resolved_commit"
        )
        != expected_commit
    )

    repo_mismatch = (
        expected_repo
        and local_source.get("repo_id")
        != expected_repo
    )

    return not (
        commit_mismatch
        or repo_mismatch
    )


def resolve_local_base_model(
    adapter_path: Path,
    configured_base_model: str,
) -> Path:
    adapter_path = Path(adapter_path)
    source = adapter_base_model_source(
        adapter_path
    )

    candidates: list[Path] = []

    configured = Path(
        configured_base_model
    )

    if configured.is_dir():
        candidates.append(configured)

    if source is not None:
        training_local_dir = source.get(
            "training_local_dir"
        )

        if training_local_dir:
            candidates.append(
                Path(str(training_local_dir))
            )

        if (
            source.get("repo_id")
            == DEFAULT_MODEL_REPO
        ):
            candidates.append(
                DEFAULT_MODEL_DIR
            )

    seen: set[str] = set()

    for candidate in candidates:
        key = str(candidate)

        if key in seen:
            continue

        seen.add(key)

        if (
            candidate.is_dir()
            and (
                source is None
                or _source_matches(
                    candidate,
                    source,
                )
            )
        ):
            return candidate

    if source and source.get("repo_id"):
        repo_id = str(source["repo_id"])

        revision = str(
            source.get("resolved_commit")
            or source.get(
                "requested_revision"
            )
            or DEFAULT_MODEL_REVISION
        )

        output_dir = (
            DEFAULT_MODEL_DIR
            if repo_id
            == DEFAULT_MODEL_REPO
            else Path("models")
            / repo_id.rsplit("/", 1)[-1]
        )

        raise FileNotFoundError(
            "Adapter base model is not "
            "available locally. This project "
            "does not download models during "
            "train/eval. Run:\n"
            "  uv run --locked --no-sync "
            "posttrain-math model download "
            f"--repo-id {repo_id} "
            f"--revision {revision} "
            f"--output-dir {output_dir}"
        )

    raise FileNotFoundError(
        "Adapter base model is not available "
        "locally and the adapter has no "
        f"{BASE_MODEL_SOURCE_FILENAME} "
        "canonical source manifest."
    )


def normalize_adapter_artifact(
    adapter_dir: Path,
    base_source: dict[str, Any],
) -> bool:
    adapter_dir = Path(adapter_dir)

    config_path = (
        adapter_dir
        / "adapter_config.json"
    )

    config = _read_json(config_path)

    if config is None:
        return False

    _write_json(
        adapter_dir
        / BASE_MODEL_SOURCE_FILENAME,
        base_source,
    )

    repo_id = base_source.get("repo_id")
    commit = base_source.get(
        "resolved_commit"
    )

    if repo_id:
        config[
            "base_model_name_or_path"
        ] = str(repo_id)

        if commit:
            config["revision"] = str(
                commit
            )

        _write_json(
            config_path,
            config,
        )

    return True


def normalize_adapter_run(
    run_dir: Path,
    base_source: dict[str, Any],
) -> list[Path]:
    run_dir = Path(run_dir)

    candidates = [
        path
        for path
        in run_dir.glob("checkpoint-*")
        if path.is_dir()
    ]

    final_model = (
        run_dir / "final-model"
    )

    if final_model.is_dir():
        candidates.append(final_model)

    normalized: list[Path] = []

    for candidate in candidates:
        if normalize_adapter_artifact(
            candidate,
            base_source,
        ):
            normalized.append(candidate)

    return normalized


@contextmanager
def staged_adapter_for_local_training(
    adapter_path: Path,
) -> Iterator[
    tuple[
        Path,
        dict[str, Any],
        Path,
    ]
]:
    adapter_path = Path(adapter_path)

    config = _read_json(
        adapter_path
        / "adapter_config.json"
    )

    if config is None:
        raise ValueError(
            "Missing adapter_config.json: "
            f"{adapter_path}"
        )

    base_model_path = (
        resolve_local_base_model(
            adapter_path,
            str(
                config.get(
                    "base_model_name_or_path",
                    "",
                )
            ),
        )
    )

    source = (
        adapter_base_model_source(
            adapter_path
        )
        or local_model_source(
            base_model_path
        )
    )

    with tempfile.TemporaryDirectory(
        prefix="posttrain-math-adapter-",
    ) as temp_dir:
        staged = (
            Path(temp_dir) / "adapter"
        )

        shutil.copytree(
            adapter_path,
            staged,
        )

        staged_config_path = (
            staged
            / "adapter_config.json"
        )

        staged_config = _read_json(
            staged_config_path
        )

        if staged_config is None:
            raise ValueError(
                "Missing staged adapter "
                f"config: {staged_config_path}"
            )

        staged_config[
            "base_model_name_or_path"
        ] = str(base_model_path)

        _write_json(
            staged_config_path,
            staged_config,
        )

        yield (
            staged,
            source,
            base_model_path,
        )


def _sha256(
    path: Path,
) -> str | None:
    if not path.is_file():
        return None

    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def _git_value(
    *args: str,
) -> str | None:
    try:
        result = subprocess.run(
            [
                "git",
                *args,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (
        OSError,
        subprocess.CalledProcessError,
    ):
        return None

    return result.stdout.strip()


def _git_dirty() -> bool | None:
    value = _git_value(
        "status",
        "--porcelain",
    )

    return (
        None
        if value is None
        else bool(value)
    )


def write_run_provenance(
    *,
    output_dir: Path,
    algorithm: str,
    model_path: Path,
    data_dir: Path,
    base_model_source: (
        dict[str, Any]
        | None
    ) = None,
) -> Path:
    output_dir = Path(output_dir)
    data_dir = Path(data_dir)
    model_path = Path(model_path)

    input_source = _read_json(
        model_path / "source.json"
    )

    adapter_source = (
        adapter_base_model_source(
            model_path
        )
    )

    resolved_base_source = (
        base_model_source
        or adapter_source
        or input_source
    )

    raw_dir = (
        data_dir.parent / "raw"
    )

    report = inspect_environment(
        train_dataset_path=(
            raw_dir
            / "math_train.parquet"
        ),
        test_dataset_path=(
            raw_dir
            / "math_test.parquet"
        ),
        model_path=model_path,
    )

    provenance = {
        "algorithm": algorithm,
        "code": {
            "git_commit":
                _git_value(
                    "rev-parse",
                    "HEAD",
                ),
            "git_dirty":
                _git_dirty(),
            "uv_lock_sha256":
                _sha256(
                    Path("uv.lock")
                ),
        },
        "input_model": {
            "path": str(
                model_path
            ),
            "source":
                input_source,
            "adapter_base_model_source":
                adapter_source,
        },
        "base_model_source":
            resolved_base_source,
        "dataset": {
            "data_dir": str(
                data_dir
            ),
            "download_manifest":
                _read_json(
                    raw_dir
                    / "download_manifest.json"
                ),
            "processed_manifest":
                _read_json(
                    data_dir
                    / "manifest.json"
                ),
        },
        "environment":
            environment_manifest(
                report
            ),
    }

    path = (
        output_dir
        / "provenance.json"
    )

    _write_json(
        path,
        provenance,
    )

    return path


def rewrite_grpo_run_config(
    *,
    run_dir: Path,
    original_adapter_path: Path,
    base_source: dict[str, Any],
) -> None:
    path = (
        Path(run_dir)
        / "run_config.json"
    )

    config = _read_json(path)

    if config is None:
        return

    config["model"] = str(
        original_adapter_path
    )

    config[
        "base_model_source"
    ] = base_source

    if base_source.get("repo_id"):
        revision = (
            base_source.get(
                "resolved_commit"
            )
            or base_source.get(
                "requested_revision"
            )
        )

        config["base_model"] = (
            f"{base_source['repo_id']}"
            f"@{revision}"
            if revision
            else str(
                base_source[
                    "repo_id"
                ]
            )
        )

    _write_json(
        path,
        config,
    )
