from __future__ import annotations

import argparse
import hashlib
import json
import math
from importlib.metadata import version
from pathlib import Path

import pandas as pd
import torch

from posttrain_math.artifacts import staged_adapter_for_local_training
from posttrain_math.prompting import PROMPT_STRATEGIES, prompt_metadata
from posttrain_math.rewards import score_boxed_numeric_completion
from posttrain_math.rl_common import (
    build_boxed_numeric_rl_dataset,
    git_commit,
    load_sft_adapter,
    resolve_rl_precision,
)

PREFIX_FRACTIONS: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _runtime_provenance() -> dict[str, str | None]:
    return {
        "git_commit": git_commit(),
        "torch": torch.__version__,
        "transformers": version("transformers"),
        "peft": version("peft"),
        "datasets": version("datasets"),
        "pandas": pd.__version__,
    }


def wilson_interval(
    successes: int,
    trials: int,
    *,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    """Two-sided Wilson score interval for a Bernoulli success probability."""
    if trials <= 0:
        raise ValueError("trials must be positive.")
    if not 0 <= successes <= trials:
        raise ValueError("successes must satisfy 0 <= successes <= trials.")
    p = successes / trials
    z2 = z * z
    denominator = 1.0 + z2 / trials
    center = (p + z2 / (2.0 * trials)) / denominator
    radius = (
        z
        * math.sqrt(p * (1.0 - p) / trials + z2 / (4.0 * trials * trials))
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def prefix_token_counts(
    completion_tokens: int,
    *,
    fractions: tuple[float, ...] = PREFIX_FRACTIONS,
) -> list[tuple[float, int]]:
    """Map fixed completion fractions to unique prefix token counts."""
    if completion_tokens <= 0:
        raise ValueError("completion_tokens must be positive.")
    output: list[tuple[float, int]] = []
    seen: set[int] = set()
    for fraction in fractions:
        if not 0.0 <= fraction < 1.0:
            raise ValueError("prefix fractions must be in [0, 1).")
        count = math.floor(completion_tokens * fraction)
        if fraction > 0.0:
            count = max(1, count)
        count = min(count, completion_tokens - 1)
        if count in seen:
            continue
        seen.add(count)
        output.append((fraction, count))
    return output


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _generate_completion_token_lists(
    *,
    model,
    tokenizer,
    state_ids: torch.Tensor,
    prompt_token_count: int,
    num_return_sequences: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
) -> list[list[int]]:
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive.")
    if num_return_sequences <= 0:
        raise ValueError("num_return_sequences must be positive.")
    _seed_everything(seed)
    attention_mask = torch.ones_like(state_ids)
    with torch.inference_mode():
        generated = model.generate(
            input_ids=state_ids,
            attention_mask=attention_mask,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=0,
            repetition_penalty=1.0,
            num_return_sequences=num_return_sequences,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )
    return [row[prompt_token_count:].tolist() for row in generated]


def _decode(tokenizer, token_ids: list[int]) -> str:
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def pre_box_token_limit(tokenizer, token_ids: list[int]) -> int:
    r"""Return a conservative token boundary before the first \boxed marker.

    The marker can span multiple tokenizer pieces. We therefore align token-prefix
    decodes against the character position where ``\boxed`` starts and keep only
    prefixes whose decoded text ends before that position. This prevents a state
    such as a trailing ``\bo``/``\box`` from leaking part of the answer marker.
    """
    full_text = _decode(tokenizer, token_ids)
    marker_start = full_text.find(r"\boxed")
    if marker_start < 0:
        return len(token_ids)

    safe_count = 0
    for end in range(1, len(token_ids) + 1):
        prefix_text = _decode(tokenizer, token_ids[:end])
        if len(prefix_text) > marker_start:
            break
        if full_text.startswith(prefix_text):
            safe_count = end
    return safe_count


def _load_selection_contract(
    selection_path: Path,
    *,
    split: str,
    prompt_name: str,
    max_completion_length: int,
    temperature: float,
    top_p: float,
) -> tuple[dict[str, object], Path]:
    config_path = selection_path.parent / "probe_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            "Frontier selection must be accompanied by its prompt-success "
            f"probe_config.json: {config_path}"
        )
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid selection probe config: {config_path}") from exc
    if not isinstance(config, dict):
        raise TypeError(f"Selection probe config must be a JSON object: {config_path}")

    expected = {
        "probe": "prompt-success-v1",
        "split": split,
        "cohort": "boxed-numeric-v1",
        "prompt": prompt_name,
        "prompt_contract": prompt_metadata(),
        "max_completion_length": max_completion_length,
    }
    mismatches = [
        f"{key}: {config.get(key)!r} != {value!r}"
        for key, value in expected.items()
        if config.get(key) != value
    ]

    sampling = config.get("sampling")
    if not isinstance(sampling, dict):
        mismatches.append("sampling: missing or not an object")
    else:
        for key, value in (("temperature", temperature), ("top_p", top_p)):
            try:
                actual = float(sampling.get(key))
            except (TypeError, ValueError):
                mismatches.append(f"sampling.{key}: missing or non-numeric")
            else:
                if not math.isclose(actual, value, rel_tol=0.0, abs_tol=1e-12):
                    mismatches.append(f"sampling.{key}: {actual!r} != {value!r}")
        if sampling.get("top_k") != 0:
            mismatches.append(f"sampling.top_k: {sampling.get('top_k')!r} != 0")

    if mismatches:
        details = "\n".join(f"- {item}" for item in mismatches)
        raise ValueError(
            "Prefix-value probe sampling/data contract does not match the "
            f"frontier selection artifact:\n{details}"
        )
    return config, config_path


def _load_probe_policy(
    *,
    staged_model: Path,
    precision: str,
):
    resolved_precision = resolve_rl_precision(precision)
    model, tokenizer, base_model_path = load_sft_adapter(
        staged_model,
        precision=resolved_precision,
        trainable=False,
    )
    model.eval()
    device = torch.device("cuda")
    model.to(device)
    return model, tokenizer, base_model_path, resolved_precision, device


def probe_prompt_success(
    *,
    model_path: Path,
    data_dir: Path,
    split: str,
    prompt_name: str,
    output_dir: Path,
    rollouts: int,
    limit_prompts: int | None,
    max_prompt_length: int,
    max_completion_length: int,
    temperature: float,
    top_p: float,
    precision: str,
    seed: int,
) -> Path:
    """Estimate p(x)=P(R=1|x, policy) for a fixed prompt set."""
    if rollouts <= 0:
        raise ValueError("rollouts must be positive.")
    if max_completion_length <= 0:
        raise ValueError("max_completion_length must be positive.")
    output_dir.mkdir(parents=True, exist_ok=True)

    with staged_adapter_for_local_training(model_path) as (
        staged_model,
        base_source,
        _base_model_path,
    ):
        model, tokenizer, base_model_path, resolved_precision, device = _load_probe_policy(
            staged_model=staged_model,
            precision=precision,
        )
        dataset, stats = build_boxed_numeric_rl_dataset(
            data_dir=data_dir,
            tokenizer=tokenizer,
            prompt_name=prompt_name,
            max_prompt_length=max_prompt_length,
            limit_prompts=limit_prompts,
            split=split,
        )

        rows: list[dict[str, object]] = []
        branch_rows: list[dict[str, object]] = []
        for prompt_index, record in enumerate(dataset):
            encoded = tokenizer(
                record["prompt"],
                add_special_tokens=False,
                return_tensors="pt",
            )
            prompt_ids = encoded["input_ids"].to(device)
            prompt_token_count = int(prompt_ids.shape[1])
            completions = _generate_completion_token_lists(
                model=model,
                tokenizer=tokenizer,
                state_ids=prompt_ids,
                prompt_token_count=prompt_token_count,
                num_return_sequences=rollouts,
                max_new_tokens=max_completion_length,
                temperature=temperature,
                top_p=top_p,
                seed=seed + prompt_index * 1009,
            )
            rewards: list[int] = []
            for rollout_index, completion_ids in enumerate(completions):
                completion = _decode(tokenizer, completion_ids)
                reward = int(
                    score_boxed_numeric_completion(
                        completion,
                        int(record["gt_numerator"]),
                        int(record["gt_denominator"]),
                    )
                )
                rewards.append(reward)
                branch_rows.append(
                    {
                        "prompt_id": record["prompt_id"],
                        "rollout_index": rollout_index,
                        "completion": completion,
                        "reward": reward,
                    }
                )
            successes = sum(rewards)
            low, high = wilson_interval(successes, rollouts)
            rows.append(
                {
                    "prompt_id": record["prompt_id"],
                    "problem": record["problem"],
                    "type": record["type"],
                    "level": record["level"],
                    "gt_numerator": int(record["gt_numerator"]),
                    "gt_denominator": int(record["gt_denominator"]),
                    "rollouts": rollouts,
                    "successes": successes,
                    "p_hat": successes / rollouts,
                    "ci_low": low,
                    "ci_high": high,
                }
            )

        output_path = output_dir / "prompt_success.parquet"
        branches_path = output_dir / "prompt_success_branches.parquet"
        pd.DataFrame(rows).to_parquet(output_path, index=False)
        pd.DataFrame(branch_rows).to_parquet(branches_path, index=False)
        config = {
            "probe": "prompt-success-v1",
            "runtime": _runtime_provenance(),
            "model": str(model_path),
            "base_model": str(base_model_path),
            "base_model_source": base_source,
            "data_dir": str(data_dir),
            "split": split,
            "cohort": "boxed-numeric-v1",
            "prompt": prompt_name,
            "prompt_contract": prompt_metadata(),
            "data": stats,
            "rollouts": rollouts,
            "max_prompt_length": max_prompt_length,
            "max_completion_length": max_completion_length,
            "sampling": {"temperature": temperature, "top_p": top_p, "top_k": 0},
            "precision": resolved_precision,
            "seed": seed,
            "output": str(output_path),
            "branches": str(branches_path),
        }
        (output_dir / "probe_config.json").write_text(
            json.dumps(config, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )
        return output_path


def probe_prefix_values(
    *,
    model_path: Path,
    data_dir: Path,
    split: str,
    prompt_name: str,
    selection_path: Path,
    output_dir: Path,
    min_p: float,
    max_p: float,
    limit_prompts: int,
    trajectories_per_prompt: int,
    branch_rollouts: int,
    max_prompt_length: int,
    max_completion_length: int,
    temperature: float,
    top_p: float,
    precision: str,
    seed: int,
) -> tuple[Path, Path, Path]:
    """Estimate Monte Carlo continuation value at fixed prefixes of sampled traces."""
    if not 0.0 <= min_p <= max_p <= 1.0:
        raise ValueError("Require 0 <= min_p <= max_p <= 1.")
    if min(limit_prompts, trajectories_per_prompt, branch_rollouts) <= 0:
        raise ValueError(
            "limit_prompts, trajectories_per_prompt and branch_rollouts must be positive."
        )
    if not selection_path.is_file():
        raise FileNotFoundError(f"Prompt selection file not found: {selection_path}")

    selection_config, selection_config_path = _load_selection_contract(
        selection_path,
        split=split,
        prompt_name=prompt_name,
        max_completion_length=max_completion_length,
        temperature=temperature,
        top_p=top_p,
    )
    selection = pd.read_parquet(selection_path)
    required_selection = {
        "prompt_id",
        "p_hat",
        "problem",
        "gt_numerator",
        "gt_denominator",
    }
    missing = required_selection - set(selection.columns)
    if missing:
        raise ValueError(f"Selection file missing columns: {sorted(missing)}")
    frontier = selection[
        selection["p_hat"].between(min_p, max_p, inclusive="both")
    ].copy()
    frontier["_distance_to_half"] = (frontier["p_hat"] - 0.5).abs()
    frontier = frontier.sort_values(
        ["_distance_to_half", "prompt_id"],
        kind="stable",
    ).drop(columns=["_distance_to_half"])
    frontier = frontier.head(limit_prompts)
    if frontier.empty:
        raise ValueError("No frontier prompts match the requested p_hat range.")
    selected_ids = set(frontier["prompt_id"].astype(str))

    output_dir.mkdir(parents=True, exist_ok=True)
    with staged_adapter_for_local_training(model_path) as (
        staged_model,
        base_source,
        _base_model_path,
    ):
        model, tokenizer, base_model_path, resolved_precision, device = _load_probe_policy(
            staged_model=staged_model,
            precision=precision,
        )
        dataset, stats = build_boxed_numeric_rl_dataset(
            data_dir=data_dir,
            tokenizer=tokenizer,
            prompt_name=prompt_name,
            max_prompt_length=max_prompt_length,
            limit_prompts=None,
            split=split,
        )
        records = [record for record in dataset if str(record["prompt_id"]) in selected_ids]
        record_by_id = {str(record["prompt_id"]): record for record in records}
        for row in frontier.itertuples(index=False):
            prompt_id = str(row.prompt_id)
            record = record_by_id.get(prompt_id)
            if record is None:
                continue
            selection_identity = (
                str(row.problem),
                int(row.gt_numerator),
                int(row.gt_denominator),
            )
            dataset_identity = (
                str(record["problem"]),
                int(record["gt_numerator"]),
                int(record["gt_denominator"]),
            )
            if selection_identity != dataset_identity:
                raise ValueError(
                    "Frontier selection identity does not match the requested "
                    f"processed dataset for {prompt_id}: "
                    f"{selection_identity!r} != {dataset_identity!r}"
                )

        ordered_records = [
            record_by_id[prompt_id]
            for prompt_id in frontier["prompt_id"].astype(str)
            if prompt_id in record_by_id
        ]
        if len(ordered_records) != len(frontier):
            missing_ids = sorted(selected_ids - set(record_by_id))
            raise ValueError(
                "Selection contains prompt IDs not present in the requested split: "
                f"{missing_ids[:5]}"
            )

        trajectory_rows: list[dict[str, object]] = []
        prefix_rows: list[dict[str, object]] = []
        branch_rows: list[dict[str, object]] = []

        for prompt_index, record in enumerate(ordered_records):
            encoded = tokenizer(
                record["prompt"],
                add_special_tokens=False,
                return_tensors="pt",
            )
            prompt_ids = encoded["input_ids"].to(device)
            prompt_token_count = int(prompt_ids.shape[1])

            for trajectory_index in range(trajectories_per_prompt):
                trajectory_seed = seed + prompt_index * 100003 + trajectory_index * 1009
                sampled = _generate_completion_token_lists(
                    model=model,
                    tokenizer=tokenizer,
                    state_ids=prompt_ids,
                    prompt_token_count=prompt_token_count,
                    num_return_sequences=1,
                    max_new_tokens=max_completion_length,
                    temperature=temperature,
                    top_p=top_p,
                    seed=trajectory_seed,
                )[0]
                if not sampled:
                    continue
                original_text = _decode(tokenizer, sampled)
                original_reward = int(
                    score_boxed_numeric_completion(
                        original_text,
                        int(record["gt_numerator"]),
                        int(record["gt_denominator"]),
                    )
                )
                trajectory_id = f"{record['prompt_id']}::t{trajectory_index}"
                trajectory_rows.append(
                    {
                        "prompt_id": record["prompt_id"],
                        "trajectory_id": trajectory_id,
                        "trajectory_index": trajectory_index,
                        "completion_tokens": len(sampled),
                        "pre_box_analysis_tokens": pre_box_token_limit(tokenizer, sampled),
                        "completion": original_text,
                        "terminal_reward": original_reward,
                    }
                )

                analysis_token_count = pre_box_token_limit(tokenizer, sampled)
                prefix_points = (
                    prefix_token_counts(analysis_token_count)
                    if analysis_token_count > 0
                    else [(0.0, 0)]
                )
                for prefix_fraction, prefix_count in prefix_points:
                    prefix_ids = sampled[:prefix_count]
                    if prefix_ids:
                        prefix_tensor = torch.tensor(
                            [prefix_ids],
                            dtype=prompt_ids.dtype,
                            device=device,
                        )
                        state_ids = torch.cat([prompt_ids, prefix_tensor], dim=1)
                    else:
                        state_ids = prompt_ids
                    state_token_count = int(state_ids.shape[1])
                    branch_budget = max_completion_length - prefix_count
                    branch_budget = max(1, branch_budget)
                    branch_seed = (
                        trajectory_seed
                        + 100_000_019
                        + prefix_count * 9176
                    )
                    branches = _generate_completion_token_lists(
                        model=model,
                        tokenizer=tokenizer,
                        state_ids=state_ids,
                        prompt_token_count=prompt_token_count,
                        num_return_sequences=branch_rollouts,
                        max_new_tokens=branch_budget,
                        temperature=temperature,
                        top_p=top_p,
                        seed=branch_seed,
                    )
                    rewards: list[int] = []
                    for branch_index, completion_ids in enumerate(branches):
                        completion = _decode(tokenizer, completion_ids)
                        reward = int(
                            score_boxed_numeric_completion(
                                completion,
                                int(record["gt_numerator"]),
                                int(record["gt_denominator"]),
                            )
                        )
                        rewards.append(reward)
                        branch_rows.append(
                            {
                                "prompt_id": record["prompt_id"],
                                "trajectory_id": trajectory_id,
                                "prefix_fraction": prefix_fraction,
                                "prefix_token_count": prefix_count,
                                "branch_index": branch_index,
                                "completion": completion,
                                "reward": reward,
                            }
                        )
                    successes = sum(rewards)
                    low, high = wilson_interval(successes, branch_rollouts)
                    prefix_rows.append(
                        {
                            "prompt_id": record["prompt_id"],
                            "trajectory_id": trajectory_id,
                            "prefix_fraction": prefix_fraction,
                            "prefix_token_count": prefix_count,
                            "state_token_count": state_token_count,
                            "prefix_text": _decode(tokenizer, prefix_ids),
                            "branch_rollouts": branch_rollouts,
                            "branch_successes": successes,
                            "value_hat": successes / branch_rollouts,
                            "ci_low": low,
                            "ci_high": high,
                            "original_terminal_reward": original_reward,
                        }
                    )

        trajectories_path = output_dir / "trajectories.parquet"
        prefix_values_path = output_dir / "prefix_values.parquet"
        branches_path = output_dir / "prefix_branches.parquet"
        pd.DataFrame(trajectory_rows).to_parquet(trajectories_path, index=False)
        pd.DataFrame(prefix_rows).to_parquet(prefix_values_path, index=False)
        pd.DataFrame(branch_rows).to_parquet(branches_path, index=False)
        config = {
            "probe": "mc-prefix-value-v1",
            "runtime": _runtime_provenance(),
            "definition": (
                "V_hat(prefix) = mean boxed-numeric-v1 terminal reward over "
                "fresh continuations sampled from that textual prefix"
            ),
            "claim_boundary": (
                "Measures continuation success under the policy; it does not "
                "establish hidden-state interpretability or process correctness."
            ),
            "model": str(model_path),
            "base_model": str(base_model_path),
            "base_model_source": base_source,
            "data_dir": str(data_dir),
            "split": split,
            "cohort": "boxed-numeric-v1",
            "prompt": prompt_name,
            "prompt_contract": prompt_metadata(),
            "data": stats,
            "selection": {
                "path": str(selection_path),
                "sha256": _sha256(selection_path),
                "probe_config_path": str(selection_config_path),
                "probe_config_sha256": _sha256(selection_config_path),
                "source_model": selection_config.get("model"),
                "source_rollouts": selection_config.get("rollouts"),
                "source_seed": selection_config.get("seed"),
            },
            "frontier": {"min_p": min_p, "max_p": max_p, "prompts": len(frontier)},
            "prefix_fractions": list(PREFIX_FRACTIONS),
            "prefix_domain": "pre-box textual span; explicit \\boxed marker excluded",
            "trajectories_per_prompt": trajectories_per_prompt,
            "branch_rollouts": branch_rollouts,
            "max_prompt_length": max_prompt_length,
            "max_completion_length": max_completion_length,
            "sampling": {"temperature": temperature, "top_p": top_p, "top_k": 0},
            "precision": resolved_precision,
            "seed": seed,
            "outputs": {
                "trajectories": str(trajectories_path),
                "prefix_values": str(prefix_values_path),
                "branches": str(branches_path),
            },
        }
        (output_dir / "probe_config.json").write_text(
            json.dumps(config, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )
        return trajectories_path, prefix_values_path, branches_path


def _add_shared_probe_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--split", choices=("train", "dev", "test"), default="dev")
    parser.add_argument("--prompt", choices=PROMPT_STRATEGIES, default="boxed")
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--max-completion-length", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--precision",
        choices=("auto", "bf16", "fp16", "fp32"),
        default="auto",
    )
    parser.add_argument("--seed", type=int, default=42)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="posttrain-math-reasoning-probe",
        description=(
            "Outcome-only RLVR diagnostics using prompt success probability and "
            "Monte Carlo continuation value of textual reasoning prefixes."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    success = subparsers.add_parser(
        "prompt-success",
        help="Estimate frozen-policy success probability p(x) for prompt selection.",
    )
    _add_shared_probe_args(success)
    success.add_argument("--output-dir", type=Path, required=True)
    success.add_argument("--rollouts", type=int, default=8)
    success.add_argument("--limit-prompts", type=int, default=128)

    prefix = subparsers.add_parser(
        "prefix-value",
        help="Estimate continuation success from fixed prefixes on frozen prompts.",
    )
    _add_shared_probe_args(prefix)
    prefix.add_argument("--selection", type=Path, required=True)
    prefix.add_argument("--output-dir", type=Path, required=True)
    prefix.add_argument("--min-p", type=float, default=0.125)
    prefix.add_argument("--max-p", type=float, default=0.875)
    prefix.add_argument("--limit-prompts", type=int, default=32)
    prefix.add_argument("--trajectories-per-prompt", type=int, default=1)
    prefix.add_argument("--branch-rollouts", type=int, default=8)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.command == "prompt-success":
            path = probe_prompt_success(
                model_path=args.model,
                data_dir=args.data_dir,
                split=args.split,
                prompt_name=args.prompt,
                output_dir=args.output_dir,
                rollouts=args.rollouts,
                limit_prompts=args.limit_prompts,
                max_prompt_length=args.max_prompt_length,
                max_completion_length=args.max_completion_length,
                temperature=args.temperature,
                top_p=args.top_p,
                precision=args.precision,
                seed=args.seed,
            )
            print(f"Prompt-success probe complete: {path}")
            return

        trajectories, prefix_values, branches = probe_prefix_values(
            model_path=args.model,
            data_dir=args.data_dir,
            split=args.split,
            prompt_name=args.prompt,
            selection_path=args.selection,
            output_dir=args.output_dir,
            min_p=args.min_p,
            max_p=args.max_p,
            limit_prompts=args.limit_prompts,
            trajectories_per_prompt=args.trajectories_per_prompt,
            branch_rollouts=args.branch_rollouts,
            max_prompt_length=args.max_prompt_length,
            max_completion_length=args.max_completion_length,
            temperature=args.temperature,
            top_p=args.top_p,
            precision=args.precision,
            seed=args.seed,
        )
        print("Prefix-value probe complete")
        print(f"- trajectories: {trajectories}")
        print(f"- prefix values: {prefix_values}")
        print(f"- branches: {branches}")
    except (FileNotFoundError, ValueError, RuntimeError, FloatingPointError) as exc:
        parser.exit(status=1, message=f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()

