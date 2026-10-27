import json

import pytest

from post_train_math.evaluation import adapter_metadata, score_completion
from post_train_math.models import MODELS


@pytest.mark.parametrize(
    "text,correct,boxed,numeric",
    [
        (r"Work. \boxed{1/2}", True, True, True),
        (r"\boxed{2}", False, True, True),
        (r"\boxed{x}", False, True, False),
        (r"\boxed{}", False, False, False),
        (r"\boxed{1/2} then \boxed{", False, False, False),
        ("The answer is 1/2", False, False, False),
        (r"\boxed{1} then \boxed{50\%}", True, True, True),
    ],
)
def test_accuracy_and_boxed_rates_have_distinct_meanings(text, correct, boxed, numeric):
    scores = score_completion(text, 1, 2)
    assert scores["correct"] is correct
    assert scores["boxed_output"] is boxed
    assert scores["numeric_boxed_output"] is numeric


def test_checkpoint_inference_metadata_does_not_require_final_report(tmp_path):
    checkpoint = tmp_path / "checkpoint-50"
    checkpoint.mkdir()
    (checkpoint / "adapter_config.json").write_text("{}")
    (checkpoint / "adapter_model.safetensors").write_bytes(b"weights")
    spec = MODELS["olmo2"]
    definition = {
        "model": {"repo_id": spec.repo_id, "revision": spec.revision},
        "data_manifest_sha256": "data-hash",
        "training": {
            "num_iterations": 1,
            "generation_batch_size": 32,
            "num_generations": 4,
        },
    }
    (checkpoint / "run_definition.json").write_text(json.dumps(definition))
    (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": 50}))
    report = adapter_metadata(checkpoint, spec, "data-hash")
    assert report["rollout_iterations"] == report["optimizer_steps"] == 50
    assert report["prompts_per_rollout_iteration"] == 8
    assert report["rollouts_per_prompt"] == 4
    assert not (tmp_path / "train_report.json").exists()
    with pytest.raises(ValueError, match="pinned model and data"):
        adapter_metadata(checkpoint, MODELS["granite"], "data-hash")
    with pytest.raises(ValueError, match="pinned model and data"):
        adapter_metadata(checkpoint, spec, "other-data")
    (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": 100}))
    with pytest.raises(ValueError, match="disagree"):
        adapter_metadata(checkpoint, spec, "data-hash")
