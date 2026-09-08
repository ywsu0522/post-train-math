from posttrain_math.rl import make_exact_rational_reward


def test_exact_rational_reward_is_binary_and_canonical() -> None:
    reward = make_exact_rational_reward()
    values = reward(
        [
            r"Final: \boxed{2/4}",
            r"Final: \boxed{3/4}",
            r"Final: \boxed{0.5}",
            "No boxed answer",
        ],
        [1, 1, 1, 1],
        [2, 2, 2, 2],
    )
    assert values == [1.0, 0.0, 0.0, 0.0]


def test_final_boxed_marker_controls_reward() -> None:
    reward = make_exact_rational_reward()
    values = reward(
        [r"First \boxed{1/2}; final \boxed{3/4}"],
        [1],
        [2],
    )
    assert values == [0.0]
