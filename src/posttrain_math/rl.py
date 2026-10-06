from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Literal

import torch
from trl import GRPOConfig, GRPOTrainer, RLOOConfig, RLOOTrainer

from posttrain_math.artifacts import local_model_source, normalize_adapter_artifact
from posttrain_math.distributed import (
    get_distributed_context,
    resolve_gradient_accumulation,
)
from posttrain_math.prompting import PROMPT_STRATEGIES
from posttrain_math.rewards import make_boxed_numeric_reward
from posttrain_math.rl_common import (
    build_boxed_numeric_rl_dataset,
    git_commit,
    load_sft_adapter,
    resolve_rl_precision,
)

AlgorithmName = Literal["rloo", "grpo", "dr-grpo"]
SUPPORTED_ALGORITHMS: tuple[str, ...] = ("rloo", "grpo", "dr-grpo")


@dataclass(frozen=True)
class AlgorithmContract:
    name: str
    display_name: str
    trainer_name: str
    variant: str
    advantage: str
    loss_normalization: str
    reward_scaling: str


_ALGORITHM_CONTRACTS: dict[str, AlgorithmContract] = {
    "rloo": AlgorithmContract(
        name="rloo",
        display_name="RLOO",
        trainer_name="RLOOTrainer",
        variant="on-policy-rloo",
        advantage="leave-one-out reward baseline",
        loss_normalization="sequence-level",
        reward_scaling="none",
    ),
    "grpo": AlgorithmContract(
        name="grpo",
        display_name="GRPO",
        trainer_name="GRPOTrainer",
        variant="original-grpo-loss",
        advantage="group mean baseline + group std scaling",
        loss_normalization="per-sequence length",
        reward_scaling="group",
    ),
    "dr-grpo": AlgorithmContract(
        name="dr-grpo",
        display_name="Dr.GRPO",
        trainer_name="GRPOTrainer",
        variant="dr-grpo-loss",
        advantage="group mean baseline without group std scaling",
        loss_normalization="global max-completion-length constant",
        reward_scaling="none",
    ),
}


def algorithm_contract(algorithm: str) -> AlgorithmContract:
    try:
        return _ALGORITHM_CONTRACTS[algorithm]
    except KeyError as exc:
        raise ValueError(
            f"Unknown RL algorithm: {algorithm}. "
            f"Expected one of {SUPPORTED_ALGORITHMS}."
        ) from exc


def _validate_common_args(
    *,
    max_steps: int,
    learning_rate: float,
    per_device_batch_size: int,
    num_generations: int,
    max_completion_length: int,
    temperature: float,
    top_p: float,
    beta: float,
    logging_steps: int,
    save_steps: int,
    save_total_limit: int,
) -> None:
    if max_steps <= 0:
        raise ValueError("max_steps must be positive.")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")
    if per_device_batch_size <= 0:
        raise ValueError("per_device_batch_size must be positive.")
    if num_generations <= 1:
        raise ValueError("num_generations must be greater than 1.")
    if max_completion_length <= 0:
        raise ValueError("max_completion_length must be positive.")
    if not 0.0 < temperature:
        raise ValueError("temperature must be positive.")
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in (0, 1].")
    if beta < 0.0:
        raise ValueError("beta must be non-negative.")
    if min(logging_steps, save_steps, save_total_limit) <= 0:
        raise ValueError(
            "logging_steps, save_steps and save_total_limit must be positive."
        )


def _common_trainer_kwargs(
    *,
    output_dir: Path,
    max_steps: int,
    learning_rate: float,
    per_device_batch_size: int,
    gradient_accumulation: int,
    precision: str,
    gradient_checkpointing: bool,
    logging_steps: int,
    save_steps: int,
    save_total_limit: int,
    seed: int,
    world_size: int,
) -> dict[str, object]:
    return {
        "output_dir": str(output_dir),
        "max_steps": max_steps,
        "learning_rate": learning_rate,
        "per_device_train_batch_size": per_device_batch_size,
        "gradient_accumulation_steps": gradient_accumulation,
        "max_grad_norm": 1.0,
        "optim": "adamw_torch",
        "bf16": precision == "bf16",
        "fp16": precision == "fp16",
        "gradient_checkpointing": gradient_checkpointing,
        "gradient_checkpointing_kwargs": (
            {"use_reentrant": False} if gradient_checkpointing else None
        ),
        "logging_strategy": "steps",
        "logging_steps": logging_steps,
        "logging_first_step": True,
        "logging_nan_inf_filter": False,
        "save_strategy": "steps",
        "save_steps": save_steps,
        "save_total_limit": save_total_limit,
        "report_to": "none",
        "seed": seed,
        "data_seed": seed,
        "disable_tqdm": True,
        "remove_unused_columns": False,
        "log_on_each_node": False,
        "ddp_find_unused_parameters": False if world_size > 1 else None,
    }


def build_trainer_config(
    *,
    algorithm: str,
    output_dir: Path,
    max_steps: int,
    learning_rate: float,
    per_device_batch_size: int,
    gradient_accumulation: int,
    num_generations: int,
    max_completion_length: int,
    temperature: float,
    top_p: float,
    beta: float,
    precision: str,
    gradient_checkpointing: bool,
    logging_steps: int,
    save_steps: int,
    save_total_limit: int,
    seed: int,
    world_size: int,
):
    """Build an explicit TRL config; no algorithm-defining default is implicit."""
    contract = algorithm_contract(algorithm)
    common = _common_trainer_kwargs(
        output_dir=output_dir,
        max_steps=max_steps,
        learning_rate=learning_rate,
        per_device_batch_size=per_device_batch_size,
        gradient_accumulation=gradient_accumulation,
        precision=precision,
        gradient_checkpointing=gradient_checkpointing,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        seed=seed,
        world_size=world_size,
    )
    generation = {
        "num_generations": num_generations,
        "num_iterations": 1,
        "max_completion_length": max_completion_length,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": 0,
        "beta": beta,
        "disable_dropout": True,
        "use_vllm": False,
    }

    if contract.name == "rloo":
        config = RLOOConfig(
            **common,
            **generation,
            epsilon=0.2,
            normalize_advantages=False,
        )
        if config.num_iterations != 1 or config.normalize_advantages:
            raise RuntimeError("RLOO semantic contract drifted from the configured baseline.")
        return config

    loss_type = "grpo" if contract.name == "grpo" else "dr_grpo"
    scale_rewards = "group" if contract.name == "grpo" else "none"
    config = GRPOConfig(
        **common,
        **generation,
        scale_rewards=scale_rewards,
        loss_type=loss_type,
    )
    if config.num_iterations != 1:
        raise RuntimeError("GRPO semantic contract requires num_iterations=1.")
    if config.loss_type != loss_type or config.scale_rewards != scale_rewards:
        raise RuntimeError("GRPO semantic contract drifted from the configured baseline.")
    return config


def train_rl(
    *,
    algorithm: str,
    model_path: Path,
    data_dir: Path,
    prompt_name: str,
    output_dir: Path,
    max_steps: int,
    learning_rate: float,
    per_device_batch_size: int,
    gradient_accumulation: int | None,
    global_batch_size: int,
    num_generations: int,
    max_prompt_length: int,
    max_completion_length: int,
    temperature: float,
    top_p: float,
    beta: float,
    precision: str,
    gradient_checkpointing: bool,
    logging_steps: int,
    save_steps: int,
    save_total_limit: int,
    seed: int,
    limit_prompts: int | None,
    resume_from_checkpoint: Path | None,
    audit_rollouts: bool = False,
    stop_after_steps: int | None = None,
) -> None:
    contract = algorithm_contract(algorithm)
    _validate_common_args(
        max_steps=max_steps,
        learning_rate=learning_rate,
        per_device_batch_size=per_device_batch_size,
        num_generations=num_generations,
        max_completion_length=max_completion_length,
        temperature=temperature,
        top_p=top_p,
        beta=beta,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
    )

    context = get_distributed_context()
    precision = resolve_rl_precision(precision)
    resolved_accumulation, effective_batch = resolve_gradient_accumulation(
        per_device_batch_size=per_device_batch_size,
        world_size=context.world_size,
        gradient_accumulation=gradient_accumulation,
        global_batch_size=global_batch_size,
    )
    if effective_batch % num_generations != 0:
        raise ValueError(
            f"{contract.display_name} effective global batch must be divisible by "
            f"num_generations: {effective_batch} % {num_generations} != 0"
        )
    prompt_groups_per_update = effective_batch // num_generations

    model, tokenizer, base_model_path = load_sft_adapter(
        model_path,
        precision=precision,
        trainable=True,
    )
    train_dataset, data_stats = build_boxed_numeric_rl_dataset(
        data_dir=data_dir,
        tokenizer=tokenizer,
        prompt_name=prompt_name,
        max_prompt_length=max_prompt_length,
        limit_prompts=limit_prompts,
        split="train",
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer_config = build_trainer_config(
        algorithm=algorithm,
        output_dir=output_dir,
        max_steps=max_steps,
        learning_rate=learning_rate,
        per_device_batch_size=per_device_batch_size,
        gradient_accumulation=resolved_accumulation,
        num_generations=num_generations,
        max_completion_length=max_completion_length,
        temperature=temperature,
        top_p=top_p,
        beta=beta,
        precision=precision,
        gradient_checkpointing=gradient_checkpointing,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        seed=seed,
        world_size=context.world_size,
    )

    algorithm_settings: dict[str, object]
    if algorithm == "rloo":
        algorithm_settings = {
            "num_iterations": trainer_config.num_iterations,
            "epsilon": trainer_config.epsilon,
            "normalize_advantages": trainer_config.normalize_advantages,
        }
    else:
        algorithm_settings = {
            "num_iterations": trainer_config.num_iterations,
            "scale_rewards": trainer_config.scale_rewards,
            "loss_type": trainer_config.loss_type,
        }

    run_config = {
        "algorithm": contract.display_name,
        "algorithm_id": contract.name,
        "variant": contract.variant,
        "git_commit": git_commit(),
        "backend": {
            "library": "trl",
            "version": version("trl"),
            "trainer": contract.trainer_name,
        },
        "model": str(model_path),
        "base_model": str(base_model_path),
        "data_dir": str(data_dir),
        "cohort": "boxed-numeric-v1",
        "prompt": prompt_name,
        "data": data_stats,
        "max_steps": max_steps,
        "learning_rate": learning_rate,
        "per_device_batch_size": per_device_batch_size,
        "world_size": context.world_size,
        "gradient_accumulation": resolved_accumulation,
        "global_effective_batch_size": effective_batch,
        "num_generations": num_generations,
        "prompt_groups_per_update": prompt_groups_per_update,
        "max_prompt_length": max_prompt_length,
        "max_completion_length": max_completion_length,
        "sampling": {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": 0,
        },
        "reward": {
            "verifier": "boxed-numeric-v1",
            "correct": 1.0,
            "otherwise": 0.0,
            "process_supervision": False,
        },
        "estimator_contract": {
            "advantage": contract.advantage,
            "loss_normalization": contract.loss_normalization,
            "reward_scaling": contract.reward_scaling,
            **algorithm_settings,
        },
        "beta": beta,
        "precision": precision,
        "gradient_checkpointing": gradient_checkpointing,
        "logging_steps": logging_steps,
        "save_steps": save_steps,
        "save_total_limit": save_total_limit,
        "seed": seed,
        "torch": torch.__version__,
        "transformers": version("transformers"),
        "trl": version("trl"),
        "peft": version("peft"),
        "datasets": version("datasets"),
    }

    if context.is_main_process:
        (output_dir / "run_config.json").write_text(
            json.dumps(run_config, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"{contract.display_name} training plan")
        print(f"- backend: TRL {version('trl')} / {contract.trainer_name}")
        print(f"- variant: {contract.variant}")
        print(f"- SFT adapter: {model_path}")
        print(f"- base model: {base_model_path}")
        print(f"- cohort prompts: {len(train_dataset)}")
        print(f"- world size: {context.world_size}")
        print(f"- per-device batch: {per_device_batch_size}")
        print(f"- gradient accumulation: {resolved_accumulation}")
        print(f"- global effective batch: {effective_batch}")
        print(f"- generations per prompt: {num_generations}")
        print(f"- prompt groups per update: {prompt_groups_per_update}")
        print(f"- max completion length: {max_completion_length}")
        print(f"- max steps: {max_steps}")
        print(f"- learning rate: {learning_rate}")
        print("- reward: boxed-numeric-v1 binary {0,1}; no process reward")
        print(f"- sampling: temperature={temperature}, top_p={top_p}")
        print(f"- KL beta: {beta}")
        print(f"- precision: {precision}")
        print()

    trainer_cls = RLOOTrainer if algorithm == "rloo" else GRPOTrainer
    callbacks = []
    reward = make_boxed_numeric_reward()
    if audit_rollouts:
        if context.world_size != 1:
            raise ValueError("Durable rollout audit currently supports a single GPU")
        from posttrain_math.rl_monitoring import RLProgressCallback, RolloutAudit

        audit = RolloutAudit(output_dir, group_size=num_generations,
                             eos_token_id=tokenizer.eos_token_id, max_tokens=max_completion_length)
        reward = audit.reward
        callbacks.append(RLProgressCallback(output_dir, audit, local_model_source(base_model_path) or {},
                                             stop_after=stop_after_steps))
    elif stop_after_steps is not None:
        raise ValueError("stop_after_steps requires rollout audit and checkpoint callbacks")
    trainer = trainer_cls(
        model=model,
        reward_funcs=reward,
        args=trainer_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        callbacks=callbacks,
    )
    train_result = trainer.train(
        resume_from_checkpoint=(
            str(resume_from_checkpoint)
            if resume_from_checkpoint is not None
            else None
        )
    )

    if trainer.state.global_step < max_steps:
        print(f"Paused cleanly at step {trainer.state.global_step}; resume the complete checkpoint.", flush=True)
        return
    final_model_dir = output_dir / "final-model"
    trainer.save_model(str(final_model_dir))
    trainer.accelerator.wait_for_everyone()

    if context.is_main_process:
        tokenizer.save_pretrained(final_model_dir)
        normalize_adapter_artifact(final_model_dir, local_model_source(base_model_path) or {})
        checkpoints = sorted(
            (
                path.name
                for path in output_dir.glob("checkpoint-*")
                if path.is_dir()
            ),
            key=lambda name: int(name.rsplit("-", 1)[1]),
        )
        summary = {
            "train_metrics": train_result.metrics,
            "checkpoints": checkpoints,
            "final_model": str(final_model_dir),
            "log_history": trainer.state.log_history,
        }
        (output_dir / "train_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )
        print()
        print(f"{contract.display_name} complete")
        print(f"- checkpoints: {len(checkpoints)}")
        print(f"- final model: {final_model_dir}")


def build_parser(*, fixed_algorithm: str | None = None) -> argparse.ArgumentParser:
    if fixed_algorithm is not None:
        algorithm_contract(fixed_algorithm)
    parser = argparse.ArgumentParser(
        prog="posttrain-math-rl",
        description=(
            "Controlled boxed-numeric RLVR using explicit TRL-backed RLOO, "
            "original GRPO, or Dr.GRPO semantics."
        ),
    )
    if fixed_algorithm is None:
        parser.add_argument("--algorithm", choices=SUPPORTED_ALGORITHMS, required=True)
    else:
        parser.set_defaults(algorithm=fixed_algorithm)
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="LoRA SFT checkpoint or final-model directory.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--prompt", choices=PROMPT_STRATEGIES, default="boxed")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=None,
        help=(
            "Explicit per-process gradient accumulation. If omitted, "
            "--global-batch-size is preserved across GPU counts."
        ),
    )
    parser.add_argument("--global-batch-size", type=int, default=4)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--max-completion-length", type=int, default=512)
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Controlled baseline uses untempered on-policy sampling.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Controlled baseline disables nucleus truncation.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.0,
        help=(
            "KL coefficient. Controlled estimator comparisons use 0.0; "
            "nonzero values intentionally change the objective."
        ),
    )
    parser.add_argument(
        "--precision",
        choices=("auto", "bf16", "fp16", "fp32"),
        default="auto",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--save-total-limit", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-prompts", type=int, default=None)
    parser.add_argument("--resume-from-checkpoint", type=Path, default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    output_dir = args.output_dir or Path(f"runs/olmo2-1b-{args.algorithm}-v1")
    try:
        train_rl(
            algorithm=args.algorithm,
            model_path=args.model,
            data_dir=args.data_dir,
            prompt_name=args.prompt,
            output_dir=output_dir,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            per_device_batch_size=args.batch_size,
            gradient_accumulation=args.gradient_accumulation,
            global_batch_size=args.global_batch_size,
            num_generations=args.num_generations,
            max_prompt_length=args.max_prompt_length,
            max_completion_length=args.max_completion_length,
            temperature=args.temperature,
            top_p=args.top_p,
            beta=args.beta,
            precision=args.precision,
            gradient_checkpointing=args.gradient_checkpointing,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            save_total_limit=args.save_total_limit,
            seed=args.seed,
            limit_prompts=args.limit_prompts,
            resume_from_checkpoint=args.resume_from_checkpoint,
        )
    except (FileNotFoundError, ValueError, RuntimeError, FloatingPointError) as exc:
        parser.exit(status=1, message=f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
