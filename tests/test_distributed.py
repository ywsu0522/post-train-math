from posttrain_math.distributed import resolve_gradient_accumulation


def test_global_batch_resolves_for_single_gpu() -> None:
    accumulation, effective = resolve_gradient_accumulation(
        per_device_batch_size=2,
        world_size=1,
        gradient_accumulation=None,
        global_batch_size=16,
    )
    assert accumulation == 8
    assert effective == 16


def test_global_batch_resolves_for_dual_gpu() -> None:
    accumulation, effective = resolve_gradient_accumulation(
        per_device_batch_size=2,
        world_size=2,
        gradient_accumulation=None,
        global_batch_size=16,
    )
    assert accumulation == 4
    assert effective == 16


def test_explicit_accumulation_takes_precedence() -> None:
    accumulation, effective = resolve_gradient_accumulation(
        per_device_batch_size=2,
        world_size=2,
        gradient_accumulation=8,
        global_batch_size=16,
    )
    assert accumulation == 8
    assert effective == 32
