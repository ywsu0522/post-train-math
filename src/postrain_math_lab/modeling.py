from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, Protocol

import torch
import transformers
from peft import PeftConfig, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


class TextGenerator(Protocol):
    def iter_generate(
        self,
        prompts: list[str],
        *,
        max_new_tokens: int,
        do_sample: bool,
    ) -> Iterator[str]:
        ...

    def generate(
        self,
        prompts: list[str],
        *,
        max_new_tokens: int,
        do_sample: bool,
    ) -> list[str]:
        ...

    def count_tokens(
        self,
        texts: list[str],
    ) -> list[int]:
        ...

    def metadata(
        self,
    ) -> dict[str, Any]:
        ...


class HFModelRunner:
    def __init__(
        self,
        *,
        model_path: Path,
        model,
        tokenizer,
        device: str,
        dtype: torch.dtype,
        batch_size: int,
    ) -> None:
        self.model_path = model_path
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.dtype = dtype
        self.batch_size = batch_size

    @classmethod
    def from_pretrained(
        cls,
        model_path: Path,
        *,
        batch_size: int = 1,
    ) -> "HFModelRunner":
        if batch_size <= 0:
            raise ValueError(
                "batch_size must be positive"
            )

        if not model_path.is_dir():
            raise FileNotFoundError(
                "Model directory not found: "
                f"{model_path}"
            )

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required "
                "for evaluation"
            )

        tokenizer = (
            AutoTokenizer
            .from_pretrained(
                model_path,
                local_files_only=True,
            )
        )

        if tokenizer.pad_token_id is None:
            if (
                tokenizer.eos_token_id
                is None
            ):
                raise RuntimeError(
                    "Tokenizer has neither "
                    "pad_token nor eos_token"
                )

            tokenizer.pad_token = (
                tokenizer.eos_token
            )

        tokenizer.padding_side = "left"

        dtype = (
            torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float16
        )

        adapter_config_path = model_path / "adapter_config.json"
        if adapter_config_path.is_file():
            peft_config = PeftConfig.from_pretrained(
                model_path,
                local_files_only=True,
            )
            base_model_path = Path(peft_config.base_model_name_or_path)
            if not base_model_path.is_dir():
                raise FileNotFoundError(
                    "LoRA adapter requires its local base model directory: "
                    f"{base_model_path}"
                )

            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                local_files_only=True,
                dtype=dtype,
            )
            model = PeftModel.from_pretrained(
                base_model,
                model_path,
                local_files_only=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                local_files_only=True,
                dtype=dtype,
            )

        model.to("cuda")
        model.eval()

        return cls(
            model_path=model_path,
            model=model,
            tokenizer=tokenizer,
            device="cuda",
            dtype=dtype,
            batch_size=batch_size,
        )

    def count_tokens(
        self,
        texts: list[str],
    ) -> list[int]:
        if not texts:
            return []

        encoded = self.tokenizer(
            texts,
            add_special_tokens=False,
            padding=False,
            truncation=False,
        )

        return [
            len(input_ids)
            for input_ids
            in encoded["input_ids"]
        ]

    def iter_generate(
        self,
        prompts: list[str],
        *,
        max_new_tokens: int,
        do_sample: bool,
    ) -> Iterator[str]:
        if max_new_tokens <= 0:
            raise ValueError(
                "max_new_tokens must "
                "be positive"
            )

        for start in range(
            0,
            len(prompts),
            self.batch_size,
        ):
            batch_prompts = prompts[
                start :
                start + self.batch_size
            ]

            encoded = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=False,
            )

            input_width = (
                encoded[
                    "input_ids"
                ].shape[1]
            )

            encoded = {
                key:
                    value.to(
                        self.device
                    )
                for key, value
                in encoded.items()
            }

            with torch.inference_mode():
                output_ids = (
                    self.model.generate(
                        **encoded,
                        do_sample=do_sample,
                        max_new_tokens=(
                            max_new_tokens
                        ),
                        pad_token_id=(
                            self.tokenizer
                            .pad_token_id
                        ),
                        eos_token_id=(
                            self.tokenizer
                            .eos_token_id
                        ),
                        use_cache=True,
                    )
                )

            continuation_ids = (
                output_ids[
                    :,
                    input_width:,
                ]
            )

            batch_generations = (
                self.tokenizer
                .batch_decode(
                    continuation_ids,
                    skip_special_tokens=True,
                )
            )

            if (
                len(batch_generations)
                != len(batch_prompts)
            ):
                raise RuntimeError(
                    "Generation count "
                    "mismatch"
                )

            for generation in (
                batch_generations
            ):
                yield generation

    def generate(
        self,
        prompts: list[str],
        *,
        max_new_tokens: int,
        do_sample: bool,
    ) -> list[str]:
        return list(
            self.iter_generate(
                prompts,
                max_new_tokens=(
                    max_new_tokens
                ),
                do_sample=do_sample,
            )
        )

    def metadata(
        self,
    ) -> dict[str, Any]:
        return {
            "model_path":
                str(self.model_path),
            "backend":
                "huggingface-transformers",
            "torch":
                torch.__version__,
            "transformers":
                transformers.__version__,
            "device":
                self.device,
            "dtype":
                str(self.dtype),
            "gpu":
                torch.cuda
                .get_device_name(0),
            "batch_size":
                self.batch_size,
        }