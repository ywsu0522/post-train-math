from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


def sort_key(name: str) -> tuple[int, int | str]:
    if name.startswith("checkpoint-"):
        try:
            return (0, int(name.rsplit("-", 1)[1]))
        except ValueError:
            return (0, name)
    if name == "final-model":
        return (1, 0)
    return (2, name)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python scripts/summarize_eval_runs.py <eval-output-root>"
        )

    root = Path(sys.argv[1])
    if not root.is_dir():
        raise SystemExit(f"Directory not found: {root}")

    rows = []
    detailed = {}

    for metrics_path in root.glob("*/metrics.json"):
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        name = metrics_path.parent.name
        rows.append(
            {
                "model": name,
                "correct": metrics["num_correct"],
                "n": metrics["num_examples"],
                "accuracy": metrics["accuracy"],
                "boxed_output_rate": metrics["boxed_output_rate"],
                "parseable_output_rate": metrics["parseable_output_rate"],
            }
        )
        detailed[name] = metrics

    rows.sort(key=lambda row: sort_key(str(row["model"])))

    if not rows:
        raise SystemExit(f"No */metrics.json files found under {root}")

    csv_path = root / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "model",
                "correct",
                "n",
                "accuracy",
                "boxed_output_rate",
                "parseable_output_rate",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    json_path = root / "summary.json"
    json_path.write_text(
        json.dumps(
            {
                "rows": rows,
                "metrics": {
                    row["model"]: detailed[row["model"]]
                    for row in rows
                },
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print("Evaluation series")
    for row in rows:
        print(
            f"- {row['model']}: "
            f"{row['correct']}/{row['n']} "
            f"accuracy={row['accuracy']:.2%} "
            f"boxed={row['boxed_output_rate']:.2%} "
            f"parseable={row['parseable_output_rate']:.2%}"
        )
    print(f"- CSV:  {csv_path}")
    print(f"- JSON: {json_path}")


if __name__ == "__main__":
    main()
