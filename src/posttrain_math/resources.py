from __future__ import annotations

import json
import shutil
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

DEFAULT_MODEL_REPO = "allenai/OLMo-1B-0724-hf"
DEFAULT_MODEL_DIR = Path("models/olmo-1b-0724-hf")


def download_model(
    *,
    repo_id: str = DEFAULT_MODEL_REPO,
    output_dir: Path = DEFAULT_MODEL_DIR,
    revision: str = "main",
    force: bool = False,
) -> Path:
    """Download a Hub model once, then use the local directory everywhere else."""
    output_dir = Path(output_dir)

    if output_dir.exists() and any(output_dir.iterdir()):
        if not force:
            manifest_path = output_dir / "source.json"
            config_path = output_dir / "config.json"
            if manifest_path.is_file() and config_path.is_file():
                print("Model already present; skipping download")
                print(f"- local directory: {output_dir}")
                return output_dir
            raise FileExistsError(
                f"Model directory is non-empty but incomplete: {output_dir}. "
                "Use --force to replace it."
            )
        shutil.rmtree(output_dir)

    output_dir.parent.mkdir(parents=True, exist_ok=True)

    info = HfApi().model_info(repo_id=repo_id, revision=revision)
    commit_sha = info.sha
    if not commit_sha:
        raise RuntimeError(f"Could not resolve a commit SHA for {repo_id}@{revision}")

    snapshot_download(
        repo_id=repo_id,
        revision=commit_sha,
        local_dir=output_dir,
    )

    manifest = {
        "repo_id": repo_id,
        "requested_revision": revision,
        "resolved_commit": commit_sha,
        "local_dir": str(output_dir),
    }
    (output_dir / "source.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("Model downloaded")
    print(f"- repo: {repo_id}")
    print(f"- revision: {revision}")
    print(f"- commit: {commit_sha}")
    print(f"- local directory: {output_dir}")

    return output_dir
