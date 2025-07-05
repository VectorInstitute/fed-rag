"""HF Multimodal Model Generator"""

from typing import Any, Dict, List, Optional, Set, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pydantic import ConfigDict, Field, PrivateAttr, model_validator
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForImageTextToText,
    AutoProcessor,
    GenerationConfig,
    PreTrainedModel,
)

from fed_rag.base.generator import BaseGenerator
from fed_rag.base.tokenizer import BaseTokenizer
from fed_rag.generators.huggingface.utils import check_huggingface_installed


class HFMultimodalModelGenerator(BaseGenerator):
    model_config = ConfigDict(
        protected_namespaces=("pydantic_model_",), arbitrary_types_allowed=True
    )
    model_name: str = Field(description="HuggingFace model name or path.")
    modality_types: Set[str] = Field(default_factory=lambda: {"text", "image"})
    generation_config: "GenerationConfig" = Field(
        default_factory=GenerationConfig
    )
    load_model_kwargs: dict = Field(default_factory=dict)
    prompt_template_init: Optional[str] = Field(
        default=None, description="Optional prompt template override."
    )
    _model: PreTrainedModel = PrivateAttr()
    _processor: Any = PrivateAttr(default=None)
    _prompt_template: str = PrivateAttr(default="")

    @model_validator(mode="before")
    @classmethod
    def _check_hf_available(cls, data: Any) -> Any:
        check_huggingface_installed(cls.__name__)
        return data

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self._prompt_template = self.prompt_template_init or ""
        self._processor = AutoProcessor.from_pretrained(self.model_name)
        cfg = AutoConfig.from_pretrained(self.model_name)
        model_cls = self._detect_model_class(cfg)
        self._model = model_cls.from_pretrained(
            self.model_name, **self.load_model_kwargs
        )

    @staticmethod
    def _detect_model_class(cfg: AutoConfig) -> Any:
        if any(
            getattr(cfg, attr, None) is not None
            for attr in ("vision_config", "audio_config", "video_config")
        ):
            return AutoModelForImageTextToText
        if getattr(cfg, "architectures", None):
            if any("ImageTextToText" in arch for arch in cfg.architectures):
                return AutoModelForImageTextToText
        return AutoModel

    def _pack_messages(
        self,
        query: str | List[str],
        context: str = "",
        images: Optional[List[Union[np.ndarray, Image.Image]]] = None,
        audios: Optional[List[np.ndarray]] = None,
        videos: Optional[List[Any]] = None,
    ) -> List[Dict[str, Any]]:
        queries = [query] if isinstance(query, str) else query
        messages: List[Dict[str, Any]] = []
        for q in queries:
            content: List[Dict[str, Any]] = []
            if context:
                content.append({"type": "text", "text": context})
            if images:
                for im in images:
                    if isinstance(im, np.ndarray):
                        im = Image.fromarray(im)
                    content.append({"type": "image", "image": im})
            if audios:
                for au in audios:
                    content.append({"type": "audio", "audio": au})
            if videos:
                for vid in videos:
                    content.append({"type": "video", "video": vid})
            content.append({"type": "text", "text": q})
            messages.append({"role": "user", "content": content})
        return (
            messages if not isinstance(query, str) else messages
        )  # always return list for consistency

    def generate(
        self,
        query: str | List[str],
        context: str = "",
        images: Optional[List[Union[np.ndarray, Image.Image]]] = None,
        audios: Optional[List[np.ndarray]] = None,
        videos: Optional[List[Any]] = None,
        max_new_tokens: int = 256,
        add_generation_prompt: bool = True,
        **gen_kwargs: Any,
    ) -> str | List[str]:
        messages = self._pack_messages(query, context, images, audios, videos)
        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )
        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self._model.generate(
                **inputs, max_new_tokens=max_new_tokens, **gen_kwargs
            )
            generation = generation[:, input_len:]
        decoded: List[str] = self._processor.batch_decode(
            generation, skip_special_tokens=True
        )
        if isinstance(query, str):
            if not decoded or not isinstance(decoded[0], str):
                raise RuntimeError("batch_decode did not return valid output")
            return decoded[0]
        return decoded

    def complete(
        self, prompt: str | List[str], **kwargs: Any
    ) -> str | List[str]:
        return self.generate(
            query=prompt,
            context="",
            images=None,
            audios=None,
            videos=None,
            **kwargs,
        )

    def compute_target_sequence_proba(
        self,
        prompt: str,
        target: str,
        context: str = "",
        images: Optional[List[Union[np.ndarray, Image.Image]]] = None,
        audios: Optional[List[np.ndarray]] = None,
        videos: Optional[List[Any]] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        text = f"{context}\n\n{prompt}" if context else prompt
        full_text = text + target
        content: List[Dict[str, Any]] = []
        if context:
            content.append({"type": "text", "text": context})
        if images:
            for im in images:
                if isinstance(im, np.ndarray):
                    im = Image.fromarray(im)
                content.append({"type": "image", "image": im})
        if audios:
            for au in audios:
                content.append({"type": "audio", "audio": au})
        if videos:
            for vid in videos:
                content.append({"type": "video", "video": vid})
        content.append({"type": "text", "text": full_text})
        messages = [{"role": "user", "content": content}]
        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"]
        prompt_inputs = self._processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "text", "text": text}]}],
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        prompt_len = prompt_inputs["input_ids"].shape[-1]
        with torch.no_grad():
            outputs = self._model(**inputs)
        if not hasattr(outputs, "logits") or outputs.logits is None:
            raise RuntimeError(
                "Underlying model does not expose logits; cannot compute probabilities."
            )
        logits = outputs.logits
        target_ids = input_ids[0][prompt_len:]
        target_logits = logits[0, prompt_len - 1 : -1, :]
        log_probs = [
            F.log_softmax(target_logits[i], dim=-1)[tid].item()
            for i, tid in enumerate(target_ids)
        ]
        return torch.exp(torch.tensor(sum(log_probs)))

    @property
    def model(self) -> PreTrainedModel:
        return self._model

    @property
    def tokenizer(self) -> BaseTokenizer:
        return self._processor

    @property
    def processor(self) -> Any:
        return self._processor

    @property
    def prompt_template(self) -> str:
        return self._prompt_template

    @prompt_template.setter
    def prompt_template(self, value: str) -> None:
        self._prompt_template = value
