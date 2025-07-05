"""HF Multimodal Model Generator

A generic generator for HuggingFace multimodal chat models (text, image, audio, etc).
Compatible with models like Gemma 3n, Llama 4, Mistral 3, DeepSeek-VL, etc.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForImageTextToText,
    AutoProcessor,
    GenerationConfig,
)

from fed_rag.base.generator import BaseGenerator


class HFMultimodalModelGenerator(BaseGenerator):
    """
    Generic generator for HuggingFace multimodal chat models.
    Supports text, image, audio modalities via HF processor.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        device_map: str = "auto",
        torch_dtype: str = "auto",
        modality_types: set = {"text", "image"},
        load_model_kwargs: Optional[dict] = None,
        prompt_template: Optional[str] = None,
        generation_config: Optional[GenerationConfig] = None,
    ):
        self.model_name = model_name
        self.modality_types = modality_types
        self._device = device
        self._prompt_template = prompt_template or ""
        self.generation_config = generation_config or GenerationConfig()
        self.load_model_kwargs = load_model_kwargs or {}
        # Load processor
        self._processor = AutoProcessor.from_pretrained(model_name)
        # Detect the model class
        config = AutoConfig.from_pretrained(model_name)
        model_cls = self._detect_model_class(config)
        # Model loading options (quantization etc.)
        model_kwargs = dict(device_map=device_map)
        if torch_dtype != "auto":
            model_kwargs["torch_dtype"] = getattr(torch, torch_dtype)
        model_kwargs.update(self.load_model_kwargs)
        # Instantiate model and move to device
        self._model = model_cls.from_pretrained(model_name, **model_kwargs)
        self._model.to(device)  # Make sure model is on the right device

    @staticmethod
    def _detect_model_class(config: Any) -> Any:
        """
        Given an HF AutoConfig, return the right AutoModel class for multimodal use-case.
        """
        # You can extend this if more types are needed
        if config.model_type in ["gemma", "mistral", "image_text_to_text"]:
            return AutoModelForImageTextToText
        if hasattr(config, "architectures"):
            if any("ImageTextToText" in arch for arch in config.architectures):
                return AutoModelForImageTextToText
        # Fallback: try generic AutoModel
        return AutoModel

    def _pack_messages(
        self,
        query: Union[str, List[str]],
        context: str = "",
        images: Optional[List[Union[np.ndarray, Image.Image]]] = None,
        audios: Optional[List[np.ndarray]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Packs the chat messages into the processor-friendly block format.
        Supports batch or single query.
        """
        if isinstance(query, str):
            # Single message
            msg_content = []
            if context:
                msg_content.append({"type": "text", "text": context})
            if images:
                for img in images:
                    if isinstance(img, np.ndarray):
                        img = Image.fromarray(img)
                    msg_content.append({"type": "image", "image": img})
            if audios:
                for audio in audios:
                    msg_content.append({"type": "audio", "audio": audio})
            msg_content.append({"type": "text", "text": query})
            return [{"role": "user", "content": msg_content}]
        else:
            # Batch mode: list of queries, repeat for each
            batch_msgs = []
            for q in query:
                msg_content = []
                if context:
                    msg_content.append({"type": "text", "text": context})
                if images:
                    for img in images:
                        if isinstance(img, np.ndarray):
                            img = Image.fromarray(img)
                        msg_content.append({"type": "image", "image": img})
                if audios:
                    for audio in audios:
                        msg_content.append({"type": "audio", "audio": audio})
                msg_content.append({"type": "text", "text": q})
                batch_msgs.append({"role": "user", "content": msg_content})
            return batch_msgs

    def generate(
        self,
        query: Union[str, List[str]],
        context: str = "",
        images: Optional[List[Union[np.ndarray, Image.Image]]] = None,
        audios: Optional[List[np.ndarray]] = None,
        max_new_tokens: int = 256,
        add_generation_prompt: bool = True,
        **gen_kwargs: Any,
    ) -> Union[str, List[str]]:
        """
        Main inference API. Packs the text, images, audio into chat messages, calls model.
        Supports single or batch queries.
        """
        # Pack messages into blocks (format expected by processor)
        messages = self._pack_messages(query, context, images, audios)
        # Processor: get model inputs (tokenization, image, etc.)
        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )
        input_len = inputs["input_ids"].shape[-1]
        # Move all tensors to correct device
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self._model.device)
        # Generation (inference mode)
        with torch.inference_mode():
            generation = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                **gen_kwargs,
            )
            generation = generation[:, input_len:]
        # Decode output(s)
        decoded: List[str] = self._processor.batch_decode(
            generation, skip_special_tokens=True
        )
        if isinstance(query, str):
            if not decoded or not isinstance(decoded[0], str):
                raise RuntimeError(
                    "batch_decode did not return a non-empty list of strings"
                )
            return decoded[0]
        else:
            return decoded

    @property
    def model(self) -> Any:
        """Return the underlying HuggingFace model."""
        return self._model

    @property
    def processor(self) -> Any:
        """Return the HuggingFace processor."""
        return self._processor

    @property
    def prompt_template(self) -> str:
        """Return the prompt template."""
        return self._prompt_template

    @prompt_template.setter
    def prompt_template(self, value: str) -> None:
        self._prompt_template = value
