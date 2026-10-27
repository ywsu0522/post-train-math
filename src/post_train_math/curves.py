"""Plot saved evaluation reports without loading model weights or requiring CUDA."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def plot_reports(paths: list[Path], output_dir: Path) -> dict:
    if not paths:
        raise ValueError("Provide at least one evaluation report")
    reports = [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]
    common = None
    groups = defaultdict(list)
    for report in reports:
        test_set = report["test_set"]
        signature = (
            report["data_manifest_sha256"],
            test_set["name"],
            test_set["problem_ids_sha256"],
            report["total"],
            report["do_sample"],
            report["max_prompt_length"],
            report["max_completion_length"],
            report["metric_definitions"],
        )
        if (
            report.get("limit") is not None
            or report["total"] != test_set["sample_size"]
        ):
            raise ValueError(
                "Smoke or incomplete evaluations cannot form a learning curve"
            )
        if report["total"] < 1 or report["do_sample"] is not False:
            raise ValueError(
                "Learning curves require nonempty deterministic evaluations"
            )
        if common is not None and signature != common:
            raise ValueError(
                "Reports use different test questions, decoding budgets, or metrics"
            )
        common = signature
        step = report["rollout_iterations"]
        if type(step) is not int or step < 0:
            raise ValueError(
                "Each report must record its nonnegative rollout iteration"
            )
        for field, count in (
            ("accuracy", "correct"),
            ("boxed_output_rate", "boxed_outputs"),
            ("numeric_boxed_output_rate", "numeric_boxed_outputs"),
        ):
            value = report[field]
            if (
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 <= value <= 1
                or not math.isclose(
                    value, report[count] / report["total"], abs_tol=1e-12
                )
            ):
                raise ValueError(f"Inconsistent metric: {field}")
        groups[(report["model_key"], report["series_name"])].append(report)
    for points in groups.values():
        points.sort(key=lambda report: report["rollout_iterations"])
        if len({point["rollout_iterations"] for point in points}) != len(points):
            raise ValueError("Duplicate rollout iterations within one model/series")
        if (
            len({json.dumps(point["model_source"], sort_keys=True) for point in points})
            != 1
        ):
            raise ValueError("One series contains different base-model sources")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fields = [
        "model_key",
        "series_name",
        "rollout_iterations",
        "optimizer_steps",
        "total",
        "correct",
        "accuracy",
        "boxed_outputs",
        "boxed_output_rate",
        "numeric_boxed_outputs",
        "numeric_boxed_output_rate",
    ]
    summaries = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    try:
        with (output_dir / "scores.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for (model, name), points in sorted(groups.items()):
                for point in points:
                    writer.writerow({key: point[key] for key in fields})
                x = [point["rollout_iterations"] for point in points]
                for axis, metric in zip(
                    axes, ("accuracy", "boxed_output_rate"), strict=True
                ):
                    axis.plot(
                        x,
                        [100 * point[metric] for point in points],
                        marker="o",
                        label=f"{model} / {name}",
                    )
                first, last = points[0], points[-1]
                summaries.append(
                    {
                        "model_key": model,
                        "series_name": name,
                        "steps": x,
                        "first_accuracy": first["accuracy"],
                        "last_accuracy": last["accuracy"],
                        "accuracy_change_percentage_points": 100
                        * (last["accuracy"] - first["accuracy"]),
                        "first_boxed_output_rate": first["boxed_output_rate"],
                        "last_boxed_output_rate": last["boxed_output_rate"],
                        "boxed_output_change_percentage_points": 100
                        * (last["boxed_output_rate"] - first["boxed_output_rate"]),
                    }
                )
        for axis, title in zip(
            axes, ("Answer accuracy (%)", "Boxed-output rate (%)"), strict=True
        ):
            axis.set(xlabel="RL rollout iterations", ylabel=title, ylim=(0, 100))
            axis.grid(alpha=0.25)
            axis.legend(fontsize=8)
        fig.suptitle(
            f"{reports[0]['test_set']['name']} / {reports[0]['total']} fixed test questions"
        )
        fig.savefig(output_dir / "learning_curves.png", dpi=180)
        fig.savefig(output_dir / "learning_curves.svg")
    finally:
        plt.close(fig)
    summary = {
        "test_set": reports[0]["test_set"],
        "metric_definitions": reports[0]["metric_definitions"],
        "series": summaries,
        "source_reports": [str(Path(path).resolve()) for path in paths],
    }
    (output_dir / "analysis.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary
