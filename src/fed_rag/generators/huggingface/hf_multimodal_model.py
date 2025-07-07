"""HF Multimodal Model Generator"""

from typing import Any

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
from fed_rag.base.generator_mixins.audio import AudioModalityMixin
from fed_rag.base.generator_mixins.image import ImageModalityMixin
from fed_rag.base.generator_mixins.video import VideoModalityMixin
from fed_rag.base.tokenizer import BaseTokenizer
from fed_rag.data_structures.generator import Context, Prompt, Query
from fed_rag.generators.huggingface.utils import check_huggingface_installed


class HFMultimodalModelGenerator(
    ImageModalityMixin,
    AudioModalityMixin,
    VideoModalityMixin,
    BaseGenerator,
):
    model_config = ConfigDict(
        protected_namespaces=("pydantic_model_",), arbitrary_types_allowed=True
    )
    model_name: str = Field(description="HuggingFace model name or path.")
    modality_types: set[str] = Field(
        default_factory=lambda: {"text", "image", "audio", "video"}
    )
    generation_config: GenerationConfig = Field(
        default_factory=GenerationConfig
    )
    load_model_kwargs: dict = Field(default_factory=dict)
    prompt_template_init: str | None = Field(default=None)
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

    def to_query(self, q: str | Query | Prompt) -> Query:
        if isinstance(q, Query):
            return q
        if isinstance(q, Prompt):
            return Query(
                text=q.text,
                images=getattr(q, "images", None),
                audios=getattr(q, "audios", None),
                videos=getattr(q, "videos", None),
            )
        return Query(text=str(q))

    def to_context(self, c: str | Context | None) -> Context | None:
        if c is None or isinstance(c, Context):
            return c
        return Context(text=str(c))

    def _pack_messages(
        self,
        query: str | Query | list[str] | list[Query],
        context: str | Context | list[str] | list[Context] | None = None,
    ) -> list[dict[str, Any]]:
        queries = (
            [query] if not isinstance(query, list) else query  # type: ignore[arg-type]
        )
        queries = [self.to_query(q) for q in queries]

        if isinstance(context, list):
            contexts = [self.to_context(c) for c in context]
            if len(contexts) != len(queries):
                raise ValueError(
                    "Batch mode requires query and context to be the same length"
                )
        else:
            contexts = [self.to_context(context)] * len(queries)

        messages: list[dict[str, Any]] = []
        for q, ctx in zip(queries, contexts):
            content: list[dict[str, Any]] = []
            if ctx is not None:
                if ctx.text:
                    content.append({"type": "text", "text": ctx.text})
                for im in ctx.images or []:
                    if isinstance(im, np.ndarray):
                        im = Image.fromarray(im)
                    content.append({"type": "image", "image": im})
                for au in ctx.audios or []:
                    content.append({"type": "audio", "audio": au})
                for vid in ctx.videos or []:
                    content.append({"type": "video", "video": vid})
            for im in q.images or []:
                if isinstance(im, np.ndarray):
                    im = Image.fromarray(im)
                content.append({"type": "image", "image": im})
            for au in q.audios or []:
                content.append({"type": "audio", "audio": au})
            for vid in q.videos or []:
                content.append({"type": "video", "video": vid})
            if q.text:
                content.append({"type": "text", "text": q.text})
            messages.append({"role": "user", "content": content})
        return messages

    def generate(
        self,
        query: str | Query | list[str] | list[Query],
        context: str | Context | list[str] | list[Context] | None = None,
        max_new_tokens: int = 256,
        add_generation_prompt: bool = True,
        **gen_kwargs: Any,
    ) -> str | list[str]:
        messages = self._pack_messages(query, context)
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
        decoded: list[str] = self._processor.batch_decode(
            generation, skip_special_tokens=True
        )
        if not isinstance(query, list):
            if not decoded or not isinstance(decoded[0], str):
                raise RuntimeError("batch_decode did not return valid output")
            return decoded[0]
        return decoded

    def complete(
        self, prompt: Prompt | list[Prompt] | str | list[str], **kwargs: Any
    ) -> str | list[str]:
        if isinstance(prompt, list):
            queries = [self.to_query(p) for p in prompt]
        else:
            queries = self.to_query(prompt)
        return self.generate(query=queries, context=None, **kwargs)

    def compute_target_sequence_proba(
        self,
        prompt: Prompt | str,
        target: str,
        context: Context | str | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        q = self.to_query(prompt)
        ctx = self.to_context(context)
        base_text = q.text or ""
        full_text = base_text + target
        content: list[dict[str, Any]] = []

        if ctx:
            if ctx.text:
                content.append({"type": "text", "text": ctx.text})
            for im in ctx.images or []:
                if isinstance(im, np.ndarray):
                    im = Image.fromarray(im)
                content.append({"type": "image", "image": im})
            for au in ctx.audios or []:
                content.append({"type": "audio", "audio": au})
            for vid in ctx.videos or []:
                content.append({"type": "video", "video": vid})

        for im in q.images or []:
            if isinstance(im, np.ndarray):
                im = Image.fromarray(im)
            content.append({"type": "image", "image": im})
        for au in q.audios or []:
            content.append({"type": "audio", "audio": au})
        for vid in q.videos or []:
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
            [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": base_text}],
                }
            ],
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
