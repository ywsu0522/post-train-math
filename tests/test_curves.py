import copy
import csv
import json

import pytest

from post_train_math.curves import plot_reports
from post_train_math.evaluation import METRIC_DEFINITIONS


def example_report(step):
    return {
        "model_key": "olmo2",
        "model_source": {"repo_id": "pinned-base", "revision": "pinned-sha"},
        "series_name": "rloo",
        "data_manifest_sha256": "data-hash",
        "test_set": {
            "name": "full",
            "sample_size": 4,
            "problem_ids_sha256": "same-four-ids",
        },
        "total": 4,
        "limit": None,
        "do_sample": False,
        "max_prompt_length": 2048,
        "max_completion_length": 2048,
        "metric_definitions": METRIC_DEFINITIONS,
        "rollout_iterations": step,
        "optimizer_steps": step,
        "correct": 1 if step == 0 else 2,
        "accuracy": 0.25 if step == 0 else 0.5,
        "boxed_outputs": 3,
        "boxed_output_rate": 0.75,
        "numeric_boxed_outputs": 2,
        "numeric_boxed_output_rate": 0.5,
    }


def test_saved_reports_produce_ordered_scores_plots_and_endpoint_analysis(tmp_path):
    paths = []
    for step in [200, 50, 0, 150, 100]:
        path = tmp_path / f"step-{step}.json"
        path.write_text(json.dumps(example_report(step)))
        paths.append(path)
    output = tmp_path / "plots"
    result = plot_reports(paths, output)
    assert result["series"][0]["steps"] == [0, 50, 100, 150, 200]
    assert result["series"][0]["accuracy_change_percentage_points"] == 25
    with (output / "scores.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["rollout_iterations"]) for row in rows] == [0, 50, 100, 150, 200]
    assert (output / "learning_curves.png").read_bytes().startswith(b"\x89PNG")
    assert "<svg" in (output / "learning_curves.svg").read_text()
    assert (
        json.loads((output / "analysis.json").read_text())["series"] == result["series"]
    )


@pytest.mark.parametrize(
    "change,error",
    [
        (
            {
                "test_set": {
                    "name": "other",
                    "sample_size": 4,
                    "problem_ids_sha256": "other-ids",
                }
            },
            "different test questions",
        ),
        ({"max_completion_length": 512}, "different test questions"),
        ({"limit": 2}, "Smoke"),
        ({"rollout_iterations": None}, "rollout iteration"),
        ({"accuracy": 0.9}, "Inconsistent metric"),
        ({"rollout_iterations": 0}, "Duplicate"),
    ],
)
def test_curve_rejects_incompatible_or_ambiguous_reports(tmp_path, change, error):
    first, second = example_report(0), copy.deepcopy(example_report(50))
    second.update(change)
    paths = [tmp_path / "first.json", tmp_path / "second.json"]
    for path, report in zip(paths, (first, second), strict=True):
        path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match=error):
        plot_reports(paths, tmp_path / "plots")
    assert not (tmp_path / "plots").exists()
