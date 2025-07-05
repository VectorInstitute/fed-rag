"""Gemma 3n Generator"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    pass

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from fed_rag.base.generator import BaseGenerator


class Gemma3nGenerator(BaseGenerator):
    """
    Generator for Gemma 3n models via HuggingFace Transformers.

    Uses AutoProcessor and AutoModelForImageTextToText for inference.
    """

    _model: Any
    _processor: Any
    _device: str

    def __init__(
        self,
        model_name: str = "google/gemma-3n-e2b-it",
        device: str = "cuda",
        device_map: str = "auto",
        torch_dtype: str = "auto",
    ):
        self._processor = AutoProcessor.from_pretrained(model_name)
        self._model = AutoModelForImageTextToText.from_pretrained(
            model_name
        ).to(device)
        self._device = device

    def generate(self, query: str, context: str = "", **kwargs: Any) -> str:
        """
        Generate a response using Gemma 3n. Returns decoded string output.

        Args:
            query (str): The prompt or question for the model.
            context (str): Optional additional context.
            **kwargs: Additional kwargs are ignored.

        Returns:
            str: The generated response from the model.
        """
        text = f"{context}\n\n{query}" if context else query
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": text}],
            }
        ]
        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        input_len = inputs["input_ids"].shape[-1]
        inputs = {
            k: (
                v.to(self._model.device, dtype=torch.long)
                if k == "input_ids"
                else v.to(self._model.device)
            )
            for k, v in inputs.items()
        }
        with torch.inference_mode():
            generation = self._model.generate(
                **inputs, max_new_tokens=256, disable_compile=False
            )
            generation = generation[:, input_len:]
        decoded = self._processor.batch_decode(
            generation, skip_special_tokens=True
        )
        if not decoded or not isinstance(decoded[0], str):
            raise RuntimeError(
                "batch_decode did not return a non-empty list of strings"
            )
        return decoded[0]

    @property
    def model(self) -> Any:
        """Return the underlying HuggingFace model."""
        return self._model

    @property
    def tokenizer(self) -> Any:
        """Return the associated processor (used as tokenizer)."""
        return self._processor

    @property
    def prompt_template(self) -> str:
        """Return an empty prompt template (not used for Gemma3n)."""
        return ""

    def complete(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("Not implemented for Gemma3nGenerator.")

    def compute_target_sequence_proba(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("Not implemented for Gemma3nGenerator.")
