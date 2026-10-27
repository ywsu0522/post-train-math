"""Metrics and portable inference metadata for saved policy checkpoints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from post_train_math.answers import (
    classify_boxed_format,
    extract_final_boxed,
    extract_final_boxed_numeric,
    verify_boxed_numeric,
)

METRIC_DEFINITIONS = {
    "accuracy": "exact rational equality using the last boxed marker; parse failures count as incorrect",
    "boxed_output_rate": "last boxed marker has a balanced, nonempty braced argument, irrespective of correctness",
    "numeric_boxed_output_rate": "last boxed marker parses under the frozen boxed-numeric grammar, irrespective of correctness",
}


def score_completion(text: str, numerator: int, denominator: int) -> dict:
    return {
        "correct": verify_boxed_numeric(
            text, numerator=numerator, denominator=denominator
        ),
        "boxed_output": extract_final_boxed(text) is not None,
        "numeric_boxed_output": extract_final_boxed_numeric(text) is not None,
        "boxed_format": classify_boxed_format(text),
    }


def adapter_metadata(adapter_dir: Path | None, spec, manifest_hash: str) -> dict:
    """A complete saved checkpoint needs no end-of-training report for inference."""
    if adapter_dir is None:
        return {
            "optimizer_steps": 0,
            "rollout_iterations": 0,
            "adapter_weights_sha256": None,
        }
    adapter_dir = Path(adapter_dir)
    weights = next(
        (
            adapter_dir / name
            for name in ("adapter_model.safetensors", "adapter_model.bin")
            if (adapter_dir / name).is_file()
        ),
        None,
    )
    if weights is None or not (adapter_dir / "adapter_config.json").is_file():
        raise ValueError(f"Missing saved LoRA adapter weights/config: {adapter_dir}")
    definition_path = adapter_dir / "run_definition.json"
    if definition_path.is_file():
        definition = json.loads(definition_path.read_text(encoding="utf-8"))
        if (
            definition.get("model")
            != {"repo_id": spec.repo_id, "revision": spec.revision}
            or definition.get("data_manifest_sha256") != manifest_hash
        ):
            raise ValueError("Checkpoint does not match the pinned model and data")
        state = json.loads(
            (adapter_dir / "trainer_state.json").read_text(encoding="utf-8")
        )
        step = state["global_step"]
        if (
            adapter_dir.name.startswith("checkpoint-")
            and adapter_dir.name != f"checkpoint-{step}"
        ):
            raise ValueError("Checkpoint directory and trainer global_step disagree")
        training = definition["training"]
    else:
        report = json.loads(
            (adapter_dir.parent / "train_report.json").read_text(encoding="utf-8")
        )
        source = report["model_source"]
        if (
            source.get("repo_id") != spec.repo_id
            or source.get("requested_revision") != spec.revision
            or source.get("resolved_commit") != spec.revision
            or report.get("data_manifest_sha256") != manifest_hash
        ):
            raise ValueError("Adapter does not match the pinned model and data")
        step = report["global_step"]
        training = report["training_config"]
    if type(step) is not int or step < 1:
        raise ValueError("Adapter must record a positive integer global_step")
    result = {
        "optimizer_steps": step,
        "rollout_iterations": step if training.get("num_iterations") == 1 else None,
        "adapter_weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
    }
    group = training.get("num_generations")
    batch = training.get("generation_batch_size")
    if (
        isinstance(group, int)
        and isinstance(batch, int)
        and group > 0
        and batch % group == 0
    ):
        result["prompts_per_rollout_iteration"] = batch // group
        result["rollouts_per_prompt"] = group
    return result
