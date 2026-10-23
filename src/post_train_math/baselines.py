"""Pinned candidate checkpoints and the native chat prompts used in the baselines."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any


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
            "You are a patient and concise mathematics tutor. "
            + _OUTPUT_CONTRACT
        ),
    ),
}


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
    messages = [
        {"role": "system", "content": spec.system_prompt},
        {"role": "user", "content": f"Problem:\n{problem}"},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def validate_model_source(model_dir: Path, spec: IFTModelSpec) -> dict:
    source = json.loads((model_dir / "source.json").read_text(encoding="utf-8"))
    if (source.get("repo_id") != spec.repo_id
            or source.get("requested_revision") != spec.revision
            or source.get("resolved_commit") != spec.revision):
        raise ValueError(f"Model is not the pinned {spec.repo_id}@{spec.revision}")
    return source


def validate_processed_manifest(data_dir: Path) -> tuple[dict, str]:
    from post_train_math.data import (
        MATH_DATASET_REPO,
        MATH_DATASET_REVISION,
        MATH_TEST_SHA256,
        MATH_TRAIN_SHA256,
        assert_current_cohort,
    )

    assert_current_cohort(data_dir)
    payload = (data_dir / "manifest.json").read_bytes()
    manifest = json.loads(payload)
    if manifest.get("split") != {"seed": 42, "dev_ratio": 0.1, "stratify": ["type", "level"]}:
        raise ValueError("Expected the fixed seed=42, dev_ratio=0.1, type/level split")
    expected_source = {
        "repo_id": MATH_DATASET_REPO,
        "revision": MATH_DATASET_REVISION,
        "sha256": {"train": MATH_TRAIN_SHA256, "test": MATH_TEST_SHA256},
    }
    if manifest.get("source") != expected_source:
        raise ValueError("Prepared data lacks verified pinned-source provenance; rerun data prepare")
    for split in ("train", "dev", "test"):
        actual = sha256((data_dir / f"{split}.parquet").read_bytes()).hexdigest()
        if actual != manifest.get("processed_sha256", {}).get(split):
            raise ValueError(f"Prepared {split} SHA256 mismatch; rerun data prepare")
    return manifest, sha256(payload).hexdigest()

