"""Resumable sampling and correctness coverage without output constraints."""
from __future__ import annotations

import hashlib
import json
import random
import time
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

from posttrain_math.answers import extract_final_boxed, parse_boxed_numeric
from posttrain_math.experiment_io import write_json
from posttrain_math.rollout_records import (
    diagnose_generation,
    summarize_diagnostics,
)


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    return {"n": n, "num_numeric": sum(r["prediction_numeric"] for r in rows),
            "num_correct": sum(r["correct"] for r in rows),
            "numeric_rate": sum(r["prediction_numeric"] for r in rows) / n if n else None,
            "accuracy": sum(r["correct"] for r in rows) / n if n else None,
            "diagnostics": summarize_diagnostics(rows)}


def coverage(rows: list[dict], samples_per_question: int) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[row["question_id"]].append(row)
    if not groups or any(sorted(r["sample_index"] for r in group) != list(range(samples_per_question))
                         for group in groups.values()):
        raise ValueError("Coverage needs complete, unique sample groups")
    correct_counts = [sum(r["correct"] for r in group) for group in groups.values()]
    n = len(groups)
    return {**summarize(rows), "questions": n, "samples_per_question": samples_per_question,
            "all_wrong_groups": sum(c == 0 for c in correct_counts),
            "mixed_groups": sum(0 < c < samples_per_question for c in correct_counts),
            "all_correct_groups": sum(c == samples_per_question for c in correct_counts),
            "observed_success_at_k": sum(c > 0 for c in correct_counts) / n,
            "mixed_group_rate": sum(0 < c < samples_per_question for c in correct_counts) / n,
            "zero_reward_variance_group_rate": sum(c in (0, samples_per_question) for c in correct_counts) / n,
            "correct_count_histogram": {str(c): correct_counts.count(c) for c in range(samples_per_question + 1)},
            "reward": "1 iff final box parses numerically and equals gold; else 0",
            "scope": "Observed success across fixed repeated samples; not a guarantee of RL trainability."}


def tasks(cases: list[dict], k: int, seed: int) -> list[dict]:
    result = []
    for case in cases:
        for sample in range(k):
            task_id = fingerprint({"question_id": case["question_id"], "sample_index": sample})
            # Independent per-completion seeds make partial-run recovery reproducible.
            task_seed = int(fingerprint({"task": task_id, "seed": seed})[:8], 16) % (2**31)
            result.append({**case, "sample_index": sample, "task_id": task_id, "seed": task_seed})
    return result


def load_records(path: Path, expected: list[dict], config: dict, *, repair_tail: bool = False) -> list[dict]:
    if not path.exists():
        return []
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        if not repair_tail:
            raise ValueError(f"Interrupted JSONL tail: {path}; rerun its stage to recover")
        data = data[:data.rfind(b"\n") + 1]
        path.write_bytes(data)
        print(f"Recovered interrupted final record: {path}", flush=True)
    rows = [json.loads(line) for line in data.splitlines()]
    if len(rows) > len(expected):
        raise ValueError("Saved records exceed the fixed experiment selection")
    for row, task in zip(rows, expected):
        checksum = row.get("record_sha256")
        if checksum != fingerprint({k: v for k, v in row.items() if k != "record_sha256"}):
            raise ValueError(f"Saved record checksum mismatch: {path}")
        if (any(row.get(k) != value for k, value in task.items())
                or row["generation_config"] != config):
            raise ValueError(f"Saved record inputs/decoding config changed: {path}")
    return rows


def run_job(path: Path, expected: list[dict], config: dict, runner_factory,
            *, limit: int | None = None) -> list[dict]:
    """Append one durable result at a time; completed records are never regenerated."""
    rows = load_records(path, expected, config, repair_tail=True)
    target = len(expected) if limit is None else limit
    if not 0 < target <= len(expected):
        raise ValueError("Job limit outside the fixed selection")
    if len(rows) >= target:
        print(f"Already complete: {path.parent.name} ({target} requested answers)", flush=True)
        return rows[:target]
    runner = runner_factory()
    path.parent.mkdir(parents=True, exist_ok=True)
    initial_count = len(rows)
    started = time.monotonic()
    print(f"{path.parent.name}: resuming at answer {len(rows) + 1}/{target}", flush=True)
    with path.open("a", encoding="utf-8") as handle:
        for task in expected[len(rows):target]:
            if config["do_sample"]:
                import torch

                random.seed(task["seed"])
                torch.manual_seed(task["seed"])
                torch.cuda.manual_seed_all(task["seed"])
            result, = runner.iter_generate_records([task["prompt"]], **config)
            boxed = extract_final_boxed(result.text)
            numeric = parse_boxed_numeric(boxed)
            row = {**task, "generation_config": config, "generation": result.text,
                   "generated_token_ids": result.token_ids, "pred_boxed": boxed,
                   "prediction_numeric": numeric is not None,
                   "correct": numeric == Fraction(task["gt_numerator"], task["gt_denominator"]),
                   "generation_diagnostics": diagnose_generation(result)}
            row["record_sha256"] = fingerprint(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            rows.append(row)
            elapsed = time.monotonic() - started
            write_json(path.parent / "progress.json", {"completed": len(rows), "target": target,
                                                       "updated_unix": time.time(), "elapsed_seconds": elapsed})
            print(f"[{path.parent.name}] {len(rows)}/{target} | "
                  f"numeric={row['prediction_numeric']} correct={row['correct']} | "
                  f"stop={result.finish_reason} tokens={len(result.token_ids)} | "
                  f"ETA~{elapsed / (len(rows) - initial_count) * (target - len(rows)):.0f}s", flush=True)
    return rows[:target]
