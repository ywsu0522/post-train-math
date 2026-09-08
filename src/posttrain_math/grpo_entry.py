from __future__ import annotations

from posttrain_math.artifacts import (
    is_main_process,
    normalize_adapter_run,
    rewrite_grpo_run_config,
    staged_adapter_for_local_training,
    write_run_provenance,
)
from posttrain_math.rl import build_parser, train_grpo


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        with staged_adapter_for_local_training(args.model) as (
            staged_model,
            base_source,
            _base_model_path,
        ):
            if is_main_process():
                write_run_provenance(
                    output_dir=args.output_dir,
                    algorithm="GRPO-original",
                    model_path=args.model,
                    data_dir=args.data_dir,
                    base_model_source=base_source,
                )

            train_grpo(
                model_path=staged_model,
                data_dir=args.data_dir,
                prompt_name=args.prompt,
                output_dir=args.output_dir,
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

            if is_main_process():
                normalize_adapter_run(args.output_dir, base_source)
                rewrite_grpo_run_config(
                    run_dir=args.output_dir,
                    original_adapter_path=args.model,
                    base_source=base_source,
                )

    except (FileNotFoundError, ValueError, RuntimeError, FloatingPointError) as exc:
        parser.exit(status=1, message=f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
