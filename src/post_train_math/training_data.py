"""Prepare native-chat training prompts and their deterministic length audit."""

from __future__ import annotations

import hashlib
import json
from importlib.metadata import version
from pathlib import Path

from post_train_math.data import COHORT_NAME
from post_train_math.models import make_chat_messages

MAX_PROMPT_LENGTH = 2048
MAX_COMPLETION_LENGTH = 2048
FROZEN_TRAIN_ROWS = 5589


def _json_bytes(value) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def prepare_training_subset(
    frame,
    tokenizer,
    spec,
    data_manifest_hash: str,
    output_dir: Path,
    *,
    write_artifacts: bool = True,
) -> tuple[list[dict], dict]:
    """Audit every frozen train question using the native generation template."""
    template = tokenizer.get_chat_template()
    selected, audit, overlong = [], [], []
    seen = set()
    for row in frame.itertuples(index=False):
        problem_id = "math-" + hashlib.sha256(row.problem.encode("utf-8")).hexdigest()
        if problem_id in seen:
            raise ValueError(f"Duplicate train problem ID: {problem_id}")
        seen.add(problem_id)
        prompt = make_chat_messages(spec, row.problem)
        encoded = tokenizer.apply_chat_template(
            prompt,
            tokenize=True,
            add_generation_prompt=True,
            truncation=False,
            padding=False,
            return_dict=True,
        )
        token_ids = encoded["input_ids"]
        prompt_tokens = len(token_ids)
        included = prompt_tokens <= MAX_PROMPT_LENGTH
        entry = {
            "problem_id": problem_id,
            "prompt_tokens": prompt_tokens,
            "type": row.type,
            "level": row.level,
            "included": included,
            "prompt_token_ids_sha256": hashlib.sha256(
                _json_bytes(token_ids)
            ).hexdigest(),
        }
        audit.append(entry)
        if included:
            selected.append(
                {
                    **entry,
                    "prompt": prompt,
                    "gt_numerator": int(row.gt_numerator),
                    "gt_denominator": int(row.gt_denominator),
                }
            )
        else:
            overlong.append(entry)
    # Record the actual tokenizer state as well as the pinned checkpoint identity.
    backend = getattr(tokenizer, "backend_tokenizer", None)
    tokenizer_state = (
        backend.to_str().encode("utf-8")
        if backend is not None
        else _json_bytes(tokenizer.get_vocab())
    )
    artifacts = {
        "training_subset.jsonl": selected,
        "prompt_audit.jsonl": audit,
        "overlong_prompts.jsonl": overlong,
    }
    payloads = {
        name: b"".join(_json_bytes(row) + b"\n" for row in rows)
        for name, rows in artifacts.items()
    }
    definition = {
        "cohort": COHORT_NAME,
        "split": "train",
        "data_manifest_sha256": data_manifest_hash,
        "problem_id_definition": "math- + SHA256 of the original problem text encoded as UTF-8",
        "model": {"key": spec.key, "repo_id": spec.repo_id, "revision": spec.revision},
        "tokenization": {
            "chat_template": template,
            "chat_template_sha256": hashlib.sha256(
                template.encode("utf-8")
            ).hexdigest(),
            "system_prompt": spec.system_prompt,
            "tokenizer_class": type(tokenizer).__name__,
            "tokenizer_state_sha256": hashlib.sha256(tokenizer_state).hexdigest(),
            "special_tokens_map": tokenizer.special_tokens_map,
            "add_generation_prompt": True,
            "truncation": False,
            "padding": False,
            "transformers_version": version("transformers"),
            "tokenizers_version": version("tokenizers"),
        },
        "selection": {
            "field": "prompt_tokens",
            "operator": "<=",
            "threshold": MAX_PROMPT_LENGTH,
        },
        "counts": {
            "cohort_train": len(audit),
            "selected": len(selected),
            "overlong": len(overlong),
        },
        "files_sha256": {
            name: hashlib.sha256(payload).hexdigest()
            for name, payload in payloads.items()
        },
    }
    if write_artifacts:
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, payload in payloads.items():
            (output_dir / name).write_bytes(payload)
        (output_dir / "training_subset_manifest.json").write_bytes(
            json.dumps(definition, ensure_ascii=False, sort_keys=True, indent=2).encode(
                "utf-8"
            )
            + b"\n"
        )
    return selected, definition
