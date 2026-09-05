from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


def get_distributed_context(
    *,
    require_cuda: bool = True,
    set_device: bool = True,
) -> DistributedContext:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if rank < 0 or local_rank < 0 or world_size <= 0:
        raise ValueError(
            "Invalid distributed environment: "
            f"RANK={rank}, LOCAL_RANK={local_rank}, WORLD_SIZE={world_size}"
        )
    if rank >= world_size:
        raise ValueError(
            f"RANK={rank} must be smaller than WORLD_SIZE={world_size}"
        )

    if not torch.cuda.is_available():
        if require_cuda:
            raise RuntimeError("CUDA is required.")
        return DistributedContext(
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            device=torch.device("cpu"),
        )

    visible_gpus = torch.cuda.device_count()
    if local_rank >= visible_gpus:
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} but only {visible_gpus} CUDA device(s) are visible."
        )

    if set_device:
        torch.cuda.set_device(local_rank)

    return DistributedContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=torch.device("cuda", local_rank),
    )


def init_process_group_if_needed(context: DistributedContext) -> bool:
    if not context.is_distributed:
        return False

    if not dist.is_available():
        raise RuntimeError("torch.distributed is unavailable.")

    if dist.is_initialized():
        return False

    backend = "nccl" if context.device.type == "cuda" else "gloo"
    dist.init_process_group(
        backend=backend,
        init_method="env://",
        device_id=context.device if context.device.type == "cuda" else None,
    )
    return True


def barrier(context: DistributedContext) -> None:
    if context.is_distributed and dist.is_initialized():
        dist.barrier()


def destroy_process_group_if_owned(owned: bool) -> None:
    if owned and dist.is_initialized():
        dist.destroy_process_group()


def resolve_gradient_accumulation(
    *,
    per_device_batch_size: int,
    world_size: int,
    gradient_accumulation: int | None,
    global_batch_size: int | None,
) -> tuple[int, int]:
    if per_device_batch_size <= 0:
        raise ValueError("per_device_batch_size must be positive.")
    if world_size <= 0:
        raise ValueError("world_size must be positive.")

    if gradient_accumulation is not None:
        if gradient_accumulation <= 0:
            raise ValueError("gradient_accumulation must be positive.")
        effective_batch = (
            per_device_batch_size
            * world_size
            * gradient_accumulation
        )
        return gradient_accumulation, effective_batch

    if global_batch_size is None or global_batch_size <= 0:
        raise ValueError(
            "Set a positive global_batch_size when gradient_accumulation is omitted."
        )

    batch_per_microstep = per_device_batch_size * world_size
    if global_batch_size % batch_per_microstep != 0:
        raise ValueError(
            "global_batch_size must be divisible by "
            "per_device_batch_size * world_size: "
            f"{global_batch_size} % ({per_device_batch_size} * {world_size}) != 0"
        )

    resolved = global_batch_size // batch_per_microstep
    if resolved <= 0:
        raise ValueError("Resolved gradient accumulation must be positive.")

    return resolved, global_batch_size
