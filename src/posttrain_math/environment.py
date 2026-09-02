import platform
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class EnvironmentReport:
    python_version: str
    torch_version: str
    torch_cuda_version: str | None
    cuda_available: bool
    gpu_name: str | None
    bf16_supported: bool

    train_dataset_exists: bool
    test_dataset_exists: bool
    model_exists: bool

    @property
    def ok(self) -> bool:
        return (
            self.cuda_available
            and self.train_dataset_exists
            and self.test_dataset_exists
            and self.model_exists
        )


def inspect_environment(
    train_dataset_path: Path,
    test_dataset_path: Path,
    model_path: Path,
) -> EnvironmentReport:
    cuda_available = torch.cuda.is_available()

    gpu_name = None
    bf16_supported = False

    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
        bf16_supported = torch.cuda.is_bf16_supported()

    return EnvironmentReport(
        python_version=platform.python_version(),
        torch_version=torch.__version__,
        torch_cuda_version=torch.version.cuda,
        cuda_available=cuda_available,
        gpu_name=gpu_name,
        bf16_supported=bf16_supported,
        train_dataset_exists=train_dataset_path.is_file(),
        test_dataset_exists=test_dataset_path.is_file(),
        model_exists=model_path.is_dir(),
    )


def print_environment_report(
    report: EnvironmentReport,
    train_dataset_path: Path,
    test_dataset_path: Path,
    model_path: Path,
) -> None:
    print("Environment")
    print(f"  Python:        {report.python_version}")
    print(f"  PyTorch:       {report.torch_version}")
    print(f"  CUDA runtime:  {report.torch_cuda_version}")
    print(f"  CUDA usable:   {report.cuda_available}")
    print(f"  GPU:           {report.gpu_name}")
    print(f"  BF16:          {report.bf16_supported}")

    print()
    print("External resources")
    print(
        f"  Train dataset: {train_dataset_path} "
        f"[{report.train_dataset_exists}]"
    )
    print(
        f"  Test dataset:  {test_dataset_path} "
        f"[{report.test_dataset_exists}]"
    )
    print(
        f"  Model:         {model_path} "
        f"[{report.model_exists}]"
    )

    print()
    print(f"Status: {'OK' if report.ok else 'FAILED'}")