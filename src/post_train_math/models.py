"""Two pinned checkpoints, native chat prompts, and local model loading."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, snapshot_download


@dataclass(frozen=True)
class IFTModelSpec:
    key: str
    repo_id: str
    revision: str
    local_dir: str
    system_prompt: str


_OUTPUT_CONTRACT = (
    "Solve the given mathematics problem carefully. "
    "Explain only the steps necessary to justify the result. "
    "Finish with exactly one final answer in LaTeX \\boxed{...}. "
    "For a rational answer, put an integer or a reduced fraction p/q "
    "inside the box; do not use decimals or extra text inside the box. "
    "Do not put any earlier intermediate expressions in boxes. "
    "Stop after the final boxed answer."
)

MODELS: dict[str, IFTModelSpec] = {
    "granite": IFTModelSpec(
        key="granite",
        repo_id="ibm-granite/granite-3.3-2b-instruct",
        revision="707f574c62054322f6b5b04b6d075f0a8f05e0f0",
        local_dir="models/ift/granite-3.3-2b-instruct",
        system_prompt=(
            "You are a precise mathematical problem-solving assistant. "
            + _OUTPUT_CONTRACT
        ),
    ),
    "olmo2": IFTModelSpec(
        key="olmo2",
        repo_id="allenai/OLMo-2-0425-1B-SFT",
        revision="0d85a3d037876ce6ac7d4311d994400fc66ac27f",
        local_dir="models/ift/olmo-2-0425-1b-sft",
        system_prompt=(
            "You are a patient and concise mathematics tutor. " + _OUTPUT_CONTRACT
        ),
    ),
}


def make_chat_messages(spec: IFTModelSpec, problem: str) -> list[dict[str, str]]:
    if not problem.strip():
        raise ValueError("Problem cannot be empty")
    return [
        {"role": "system", "content": spec.system_prompt},
        {"role": "user", "content": f"Problem:\n{problem.strip()}"},
    ]


def make_chat_prompt(tokenizer: Any, spec: IFTModelSpec, problem: str) -> str:
    """Use the exact model tokenizer template, including its BOS and role tokens.

    Granite's implicit dated system prompt is bypassed by providing an
    explicit system message. We intentionally do not enable thinking=True.
    OLMo's own tokenizer decides whether to insert an initial BOS token.
    """
    problem = problem.strip()
    if not problem:
        raise ValueError("Problem cannot be empty")
    if not getattr(tokenizer, "chat_template", None):
        raise RuntimeError(f"{spec.repo_id} tokenizer has no chat template")
    messages = make_chat_messages(spec, problem)
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def validate_model_source(model_dir: Path, spec: IFTModelSpec) -> dict:
    source = json.loads((model_dir / "source.json").read_text(encoding="utf-8"))
    if (
        source.get("repo_id") != spec.repo_id
        or source.get("requested_revision") != spec.revision
        or source.get("resolved_commit") != spec.revision
    ):
        raise ValueError(f"Model is not the pinned {spec.repo_id}@{spec.revision}")
    return source


def download_model(
    *,
    repo_id: str,
    output_dir: Path,
    revision: str,
    force: bool = False,
) -> Path:
    """Download a Hub model once, then use the local directory everywhere else."""
    output_dir = Path(output_dir)

    if output_dir.exists() and any(output_dir.iterdir()):
        if not force:
            manifest_path = output_dir / "source.json"
            config_path = output_dir / "config.json"
            if manifest_path.is_file() and config_path.is_file():
                existing_source = json.loads(manifest_path.read_text(encoding="utf-8"))
                if (
                    existing_source.get("repo_id") != repo_id
                    or existing_source.get("requested_revision") != revision
                    or existing_source.get("resolved_commit") != revision
                ):
                    raise ValueError(
                        "Existing model source does not match requested repo/revision: "
                        f"{output_dir}. Use a distinct --output-dir or --force."
                    )
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
    if commit_sha != revision:
        raise RuntimeError(f"Model revision mismatch: {commit_sha} != {revision}")

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


def load_tokenizer(model_dir: Path):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer needs a pad or EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer
