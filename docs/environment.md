# GPU environment contract

## Reference runtime

This repository's reproducible GPU target is Linux x86_64, Python 3.12, and the
committed `uv.lock`. The current lock resolves `torch==2.13.0` on Linux with a
CUDA 13.x runtime dependency set.

CUDA 13 requires NVIDIA Turing or newer for the toolkit/library support used by
this project. The practical minimum is compute capability 7.5 (SM75). CUDA 13.x
minor-version compatibility requires an NVIDIA R580+ driver.

References:

- NVIDIA CUDA 13.0 release notes:
  https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/
- uv PyTorch integration:
  https://docs.astral.sh/uv/guides/integration/pytorch/

## Compatibility policy

The project checks capabilities instead of cloud-provider names.

Required for GPU training/evaluation:

1. `torch.cuda.is_available()` is true.
2. `torch.version.cuda` reports CUDA 13.x for the reference lock.
3. `nvidia-smi` reports driver major version 580 or newer.
4. Every visible training GPU has compute capability >= 7.5.

T4 / SM75 runs in FP16. `--precision auto` selects BF16 only when PyTorch
reports native BF16 support.

Run:

```bash
uv run --locked --no-sync posttrain-math doctor
```

To save a machine-readable report:

```bash
uv run --locked --no-sync posttrain-math doctor \
  --output runs/environment.json
```

`doctor` also prints `torch.cuda.get_arch_list()` so a reviewer can see the CUDA
architectures compiled into the installed Torch binary.

## Multi-GPU

`scripts/launch_gpu.sh` and `scripts/launch_grpo.sh` use `torchrun` with one
process per selected GPU. The reference distributed backend is NCCL on Linux.

A Windows CUDA installation may work for single-GPU Python execution, but the
portfolio's tested/reference multi-GPU contract is Linux + NCCL. macOS can run
CPU-side data/verifier tests but is outside the GPU training contract.

## Changing the Torch/CUDA build

Treat a Torch/CUDA change as an environment-contract change:

1. edit `pyproject.toml` if the Torch source/version changes;
2. regenerate `uv.lock`;
3. update the constants/checks in `posttrain_math.environment`;
4. rerun unit tests and `doctor` on a real GPU;
5. update this document with the new driver and architecture floor.

Do not add Colab/Kaggle conditionals to core Python code. Hosted platforms are
deployment procedures around the same hardware contract.
