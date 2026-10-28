"""Adapter, AdamW and RNG checkpoint I/O, independent of the RLOO algorithm."""

from __future__ import annotations

import json
from pathlib import Path

ENGINE = "explicit-rloo-v1"


def run_definition(spec, manifest_hash, config, lora):
    training = config.to_dict()
    for key in ("output_dir", "max_steps"):
        training.pop(key)
    lora_settings = lora.to_dict()
    lora_settings["target_modules"] = sorted(lora.target_modules)
    return {
        "engine": ENGINE,
        "model": {"repo_id": spec.repo_id, "revision": spec.revision},
        "data_manifest_sha256": manifest_hash,
        "training": training,
        "lora": lora_settings,
    }


def rng_filename(rank, world_size):
    return "rng_state.pth" if world_size == 1 else f"rng_state_{rank}.pth"


def validate_checkpoint(path, definition, max_steps):
    path = Path(path)
    required = [
        "adapter_model.safetensors",
        "adapter_config.json",
        "optimizer.pt",
        "trainer_state.json",
        "run_definition.json",
    ]
    required += [
        rng_filename(rank, definition["training"]["world_size"])
        for rank in range(definition["training"]["world_size"])
    ]
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise ValueError(f"Incomplete checkpoint: missing {', '.join(missing)}")
    saved = json.loads((path / "run_definition.json").read_text(encoding="utf-8"))
    if saved.get("engine") != ENGINE:
        raise ValueError("This training loop requires an explicit-rloo-v1 checkpoint")
    if saved != definition:
        raise ValueError("Checkpoint model, data, LoRA or training settings differ")
    state = json.loads((path / "trainer_state.json").read_text(encoding="utf-8"))
    if not 0 < state["global_step"] < max_steps:
        raise ValueError("max-steps must exceed the checkpoint global_step")
    return state


def save_checkpoint(
    path,
    model,
    tokenizer,
    optimizer,
    accelerator,
    definition,
    step,
    history,
    *,
    weights_only=False,
):
    import torch

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.unwrap_model(model).save_pretrained(path)
        tokenizer.save_pretrained(path)
        for filename, value in (
            ("run_definition.json", definition),
            ("trainer_state.json", {"global_step": step, "log_history": history}),
        ):
            (path / filename).write_text(
                json.dumps(value, indent=2) + "\n", encoding="utf-8"
            )
        if not weights_only:
            torch.save(optimizer.state_dict(), path / "optimizer.pt")
    if not weights_only:
        rng = {"cpu": torch.get_rng_state()}
        if accelerator.device.type == "cuda":
            rng["cuda"] = torch.cuda.get_rng_state(accelerator.device)
        if accelerator.scaler is not None:
            rng["scaler"] = accelerator.scaler.state_dict()
        torch.save(
            rng,
            path / rng_filename(accelerator.process_index, accelerator.num_processes),
        )
    accelerator.wait_for_everyone()


def load_checkpoint(path, model, optimizer, accelerator, definition, max_steps):
    import torch
    from peft import set_peft_model_state_dict
    from peft.utils.save_and_load import load_peft_weights

    path = Path(path)
    state = validate_checkpoint(path, definition, max_steps)
    set_peft_model_state_dict(
        accelerator.unwrap_model(model), load_peft_weights(path, device="cpu")
    )
    optimizer.load_state_dict(
        torch.load(
            path / "optimizer.pt", map_location=accelerator.device, weights_only=True
        )
    )
    rng = torch.load(
        path / rng_filename(accelerator.process_index, accelerator.num_processes),
        map_location="cpu",
        weights_only=True,
    )
    torch.set_rng_state(rng["cpu"])
    if accelerator.device.type == "cuda":
        torch.cuda.set_rng_state(rng["cuda"], accelerator.device)
    if accelerator.scaler is not None:
        accelerator.scaler.load_state_dict(rng["scaler"])
    return state["global_step"], state["log_history"]
