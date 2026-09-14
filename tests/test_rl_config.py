from pathlib import Path

from posttrain_math.rl import build_trainer_config


def _config(algorithm: str):
    return build_trainer_config(
        algorithm=algorithm,
        output_dir=Path("unused"),
        max_steps=1,
        learning_rate=1e-6,
        per_device_batch_size=4,
        gradient_accumulation=1,
        num_generations=4,
        max_completion_length=64,
        temperature=1.0,
        top_p=1.0,
        beta=0.0,
        precision="fp32",
        gradient_checkpointing=False,
        logging_steps=1,
        save_steps=1,
        save_total_limit=1,
        seed=42,
        world_size=1,
    )


def test_original_grpo_contract_is_explicit() -> None:
    config = _config("grpo")
    assert config.loss_type == "grpo"
    assert config.scale_rewards == "group"
    assert config.num_iterations == 1
    assert config.beta == 0.0
    assert config.temperature == 1.0
    assert config.top_p == 1.0


def test_dr_grpo_contract_removes_reward_std_scaling() -> None:
    config = _config("dr-grpo")
    assert config.loss_type == "dr_grpo"
    assert config.scale_rewards == "none"
    assert config.num_iterations == 1


def test_rloo_contract_is_one_iteration_without_advantage_normalization() -> None:
    config = _config("rloo")
    assert config.num_iterations == 1
    assert config.normalize_advantages is False
    assert config.beta == 0.0
    assert config.temperature == 1.0
    assert config.top_p == 1.0
