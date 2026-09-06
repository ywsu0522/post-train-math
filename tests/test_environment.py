from posttrain_math.environment import (
    REFERENCE_CUDA_MAJOR,
    REFERENCE_MIN_COMPUTE_CAPABILITY,
    REFERENCE_MIN_DRIVER_MAJOR,
    _driver_major,
)


def test_reference_gpu_contract_is_explicit() -> None:
    assert REFERENCE_CUDA_MAJOR == 13
    assert REFERENCE_MIN_DRIVER_MAJOR == 580
    assert REFERENCE_MIN_COMPUTE_CAPABILITY == (7, 5)


def test_driver_major_parser() -> None:
    assert _driver_major("580.82.07") == 580
    assert _driver_major("610.43.02") == 610
    assert _driver_major(None) is None
    assert _driver_major("unknown") is None
