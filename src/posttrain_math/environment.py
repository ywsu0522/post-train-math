from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

REFERENCE_CUDA_MAJOR = 13
REFERENCE_MIN_DRIVER_MAJOR = 580
REFERENCE_MIN_COMPUTE_CAPABILITY = (7, 5)


def native_bf16_supported() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported(
        including_emulation=False
    )


def _driver_version() -> str | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None

    versions = {
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip()
    }
    if len(versions) != 1:
        return None
    return versions.pop()


def _driver_major(version: str | None) -> int | None:
    if not version:
        return None
    try:
        return int(version.split(".", 1)[0])
    except ValueError:
        return None


def _compiled_cuda_arches() -> tuple[str, ...]:
    try:
        return tuple(torch.cuda.get_arch_list())
    except (AttributeError, RuntimeError):
        return ()


@dataclass(frozen=True)
class GPUReport:
    index: int
    name: str
    compute_capability: str
    total_vram_gib: float
    bf16_supported: bool
    architecture_supported: bool


@dataclass(frozen=True)
class EnvironmentReport:
    os: str
    machine: str
    python_version: str
    torch_version: str
    torch_cuda_version: str | None
    compiled_cuda_arches: tuple[str, ...]
    cuda_available: bool
    driver_version: str | None
    driver_supported: bool
    gpus: tuple[GPUReport, ...]

    train_dataset_exists: bool
    test_dataset_exists: bool
    model_exists: bool

    @property
    def hardware_ok(self) -> bool:
        return (
            self.cuda_available
            and self.driver_supported
            and bool(self.gpus)
            and all(gpu.architecture_supported for gpu in self.gpus)
        )

    @property
    def ok(self) -> bool:
        # Resource files are informational: `doctor` is useful on a fresh clone
        # before model/data acquisition.
        return self.hardware_ok


def inspect_environment(
    train_dataset_path: Path,
    test_dataset_path: Path,
    model_path: Path,
) -> EnvironmentReport:
    cuda_available = torch.cuda.is_available()
    driver_version = _driver_version() if cuda_available else None
    driver_major = _driver_major(driver_version)

    torch_cuda_version = torch.version.cuda
    cuda_major = None
    if torch_cuda_version:
        try:
            cuda_major = int(torch_cuda_version.split(".", 1)[0])
        except ValueError:
            cuda_major = None

    # The committed reference lock resolves Linux torch to CUDA 13.x.
    # Fail closed if a future dependency update changes the CUDA major without
    # updating this explicit contract.
    driver_supported = bool(
        cuda_available
        and cuda_major == REFERENCE_CUDA_MAJOR
        and driver_major is not None
        and driver_major >= REFERENCE_MIN_DRIVER_MAJOR
    )

    gpus: list[GPUReport] = []
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            major, minor = torch.cuda.get_device_capability(index)
            props = torch.cuda.get_device_properties(index)
            gpus.append(
                GPUReport(
                    index=index,
                    name=torch.cuda.get_device_name(index),
                    compute_capability=f"{major}.{minor}",
                    total_vram_gib=props.total_memory / (1024**3),
                    # Native BF16 starts with Ampere-class NVIDIA devices for
                    # the GPU families supported by this project.
                    bf16_supported=major >= 8,
                    architecture_supported=(
                        (major, minor) >= REFERENCE_MIN_COMPUTE_CAPABILITY
                    ),
                )
            )

    return EnvironmentReport(
        os=platform.system(),
        machine=platform.machine(),
        python_version=platform.python_version(),
        torch_version=torch.__version__,
        torch_cuda_version=torch_cuda_version,
        compiled_cuda_arches=_compiled_cuda_arches(),
        cuda_available=cuda_available,
        driver_version=driver_version,
        driver_supported=driver_supported,
        gpus=tuple(gpus),
        train_dataset_exists=train_dataset_path.is_file(),
        test_dataset_exists=test_dataset_path.is_file(),
        model_exists=model_path.is_dir(),
    )


def environment_manifest(report: EnvironmentReport) -> dict:
    return {
        "reference_contract": {
            "cuda_major": REFERENCE_CUDA_MAJOR,
            "minimum_driver_major": REFERENCE_MIN_DRIVER_MAJOR,
            "minimum_compute_capability": (
                f"{REFERENCE_MIN_COMPUTE_CAPABILITY[0]}."
                f"{REFERENCE_MIN_COMPUTE_CAPABILITY[1]}"
            ),
        },
        **asdict(report),
        "hardware_ok": report.hardware_ok,
    }


def write_environment_manifest(
    report: EnvironmentReport,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(environment_manifest(report), indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )


def print_environment_report(
    report: EnvironmentReport,
    train_dataset_path: Path,
    test_dataset_path: Path,
    model_path: Path,
) -> None:
    print("Reference GPU contract")
    print(f"  CUDA major:            {REFERENCE_CUDA_MAJOR}.x")
    print(f"  NVIDIA driver:         R{REFERENCE_MIN_DRIVER_MAJOR}+")
    print(
        "  Minimum GPU compute:   "
        f"{REFERENCE_MIN_COMPUTE_CAPABILITY[0]}."
        f"{REFERENCE_MIN_COMPUTE_CAPABILITY[1]} (Turing / SM75)"
    )

    print("\nSoftware")
    print(f"  OS / machine:          {report.os} / {report.machine}")
    print(f"  Python:                {report.python_version}")
    print(f"  PyTorch:               {report.torch_version}")
    print(f"  Torch CUDA build:      {report.torch_cuda_version}")
    print(
        "  Compiled CUDA archs:   "
        + (", ".join(report.compiled_cuda_arches) or "<unavailable>")
    )
    print(f"  CUDA usable:           {report.cuda_available}")
    print(f"  NVIDIA driver:         {report.driver_version}")
    print(
        "  Driver contract:       "
        f"{'PASS' if report.driver_supported else 'FAIL'}"
    )

    print("\nGPUs")
    if not report.gpus:
        print("  <none>")
    for gpu in report.gpus:
        recommended_dtype = "bf16" if gpu.bf16_supported else "fp16"
        print(
            f"  [{gpu.index}] {gpu.name} | sm={gpu.compute_capability} | "
            f"vram={gpu.total_vram_gib:.1f} GiB | "
            f"dtype={recommended_dtype} | "
            f"arch={'PASS' if gpu.architecture_supported else 'FAIL'}"
        )

    print("\nExternal resources (informational)")
    print(
        f"  Train dataset: {train_dataset_path} "
        f"[{report.train_dataset_exists}]"
    )
    print(
        f"  Test dataset:  {test_dataset_path} "
        f"[{report.test_dataset_exists}]"
    )
    print(f"  Model:         {model_path} [{report.model_exists}]")

    print()
    print(f"GPU training status: {'OK' if report.ok else 'FAILED'}")
