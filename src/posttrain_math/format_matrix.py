"""Run the four format experiments in a fixed order with shared settings."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from posttrain_math.prompting import prompt_metadata

EXPERIMENTS = (
    ("01-base-fsm-off", "base", "off"),
    ("02-base-fsm-on", "base", "on"),
    ("03-sft-fsm-off", "sft", "off"),
    ("04-sft-fsm-on", "sft", "on"),
)


def run_format_matrix(
    *, base_model: Path, sft_model: Path, data_dir: Path, output_dir: Path,
    split: str, batch_size: int, max_new_tokens: int, limit: int | None,
) -> None:
    for model in (base_model, sft_model):
        if not model.is_dir():
            raise FileNotFoundError(f"Model directory not found: {model}")
    if min(batch_size, max_new_tokens) <= 0 or (limit is not None and limit <= 0):
        raise ValueError("batch_size, max_new_tokens and limit must be positive.")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "cohort": "boxed-numeric-v1",
        "prompt_contract": prompt_metadata(),
        "split": split,
        "data_dir": str(data_dir),
        "max_new_tokens": max_new_tokens,
        "batch_size": batch_size,
        "limit": limit,
        "do_sample": False,
        "runs": [],
    }
    summary_path = output_dir / "matrix.json"
    for name, kind, fsm in EXPERIMENTS:
        model = base_model if kind == "base" else sft_model
        run_dir = output_dir / name
        command = [
            sys.executable, "-m", "posttrain_math", "eval",
            "--model", str(model), "--data-dir", str(data_dir),
            "--split", split, "--prompt", "boxed", "--fsm", fsm,
            "--batch-size", str(batch_size), "--max-new-tokens", str(max_new_tokens),
            "--output-dir", str(run_dir),
        ]
        if limit is not None:
            command.extend(["--limit", str(limit)])
        record = {"name": name, "model": str(model), "fsm": fsm, "status": "running"}
        summary["runs"].append(record)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"Starting {name}", flush=True)
        try:
            subprocess.run(command, check=True)
            metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            record.update({
                "status": "complete",
                "metrics_path": str(run_dir / "metrics.json"),
                **{key: metrics[key] for key in (
                    "num_examples", "accuracy", "boxed_output_rate",
                    "grammar_output_rate", "numeric_output_rate",
                )},
            })
        except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
            record.update({"status": "failed", "error": str(exc)})
            raise
        finally:
            summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Format matrix complete: {summary_path}")
