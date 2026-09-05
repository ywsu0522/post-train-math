from posttrain_math.rl import make_math_verify_reward


def test_math_verify_reward() -> None:
    reward = make_math_verify_reward(
        format_reward_weight=0.05,
    )

    values = reward(
        [
            r"Final: \boxed{2}",
            r"Final: \boxed{3}",
            "No boxed answer",
        ],
        ["2", "2", "2"],
    )

    assert values == [1.0, 0.05, 0.0]
