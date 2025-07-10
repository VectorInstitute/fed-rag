"""Unsloth Fast Multimodal Model Generator (aligned with HF style)"""

from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from pydantic import ConfigDict, Field, PrivateAttr, model_validator

# Expose FastModel for patching/tests (unlike HF classes, it's not always module-level).
try:
    from unsloth import FastModel
except ImportError:
    FastModel = None  # for testing/mocking

if TYPE_CHECKING:
    from unsloth import FastModel

from fed_rag.base.generator import BaseGenerator
from fed_rag.base.generator_mixins.audio import AudioModalityMixin
from fed_rag.base.generator_mixins.image import ImageModalityMixin
from fed_rag.base.generator_mixins.video import VideoModalityMixin
from fed_rag.data_structures.rag import Context, Prompt, Query
from fed_rag.exceptions.generator import GeneratorError

from .mixin import UnslothGeneratorMixin
from .utils import check_unsloth_installed


class UnslothFastMultimodalModelGenerator(
    ImageModalityMixin,
    AudioModalityMixin,
    VideoModalityMixin,
    UnslothGeneratorMixin,
    BaseGenerator,
):
    model_config = ConfigDict(
        protected_namespaces=("pydantic_model_",), arbitrary_types_allowed=True
    )
    model_name: str = Field(description="Unsloth model name or path.")
    modality_types: set[str] = Field(
        default_factory=lambda: {"text", "image", "audio", "video"}
    )
    generation_config: Optional[Any] = Field(default=None)
    load_model_kwargs: dict = Field(default_factory=dict)
    prompt_template_init: str | None = Field(default=None)
    load_model_at_init: bool = Field(default=True)

    _model: Optional["FastModel"] = PrivateAttr(default=None)
    _processor: Any = PrivateAttr(default=None)
    _prompt_template: str = PrivateAttr(default="")

    @model_validator(mode="before")
    @classmethod
    def _check_unsloth_available(cls, data: Any) -> Any:
        check_unsloth_installed(cls.__name__)
        return data

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self._prompt_template = self.prompt_template_init or ""
        self._model = None
        self._processor = None
        if self.load_model_at_init:
            self._model, self._processor = self._load_model_from_unsloth()

    def _load_model_from_unsloth(self) -> tuple[Any, Any]:
        from unsloth import FastModel

        model, processor = FastModel.from_pretrained(
            model_name=self.model_name,
            **self.load_model_kwargs,
        )
        return model, processor

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
                raise GeneratorError(
                    "Batch mode requires query and context to be the same length"
                )
        else:
            contexts = [self.to_context(context)] * len(queries)

        messages: list[dict[str, Any]] = []
        for q, ctx in zip(queries, contexts):
            content: list[dict[str, Any]] = []
            if ctx is not None:
                if getattr(ctx, "text", None):
                    content.append({"type": "text", "text": ctx.text})
                for im in getattr(ctx, "images", []) or []:
                    from PIL import Image as PILImage

                    if isinstance(im, np.ndarray):
                        im = PILImage.fromarray(im)
                    content.append({"type": "image", "image": im})
                for au in getattr(ctx, "audios", []) or []:
                    content.append({"type": "audio", "audio": au})
                for vid in getattr(ctx, "videos", []) or []:
                    content.append({"type": "video", "video": vid})
            for im in getattr(q, "images", []) or []:
                from PIL import Image as PILImage

                if isinstance(im, np.ndarray):
                    im = PILImage.fromarray(im)
                content.append({"type": "image", "image": im})
            for au in getattr(q, "audios", []) or []:
                content.append({"type": "audio", "audio": au})
            for vid in getattr(q, "videos", []) or []:
                content.append({"type": "video", "video": vid})
            if getattr(q, "text", None):
                content.append({"type": "text", "text": q.text})

            messages.append({"role": "user", "content": content})
        return messages

    def generate(
        self,
        query: str | Query | list[str] | list[Query],
        context: str | Context | list[str] | list[Context] | None = None,
        **gen_kwargs: Any,
    ) -> str | list[str]:
        max_new_tokens = gen_kwargs.pop("max_new_tokens", 256)
        add_generation_prompt = gen_kwargs.pop("add_generation_prompt", True)
        messages = self._pack_messages(query, context)
        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )

        # Unsloth: must manually move all input tensors to the model device.
        model_device = next(self.model.parameters()).device
        inputs = {
            k: v.to(model_device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            generation = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, **gen_kwargs
            )
            generation = generation[:, input_len:]
        decoded: list[str] = self._processor.batch_decode(
            generation, skip_special_tokens=True
        )
        if not isinstance(query, list):
            if not decoded or not isinstance(decoded[0], str):
                raise GeneratorError(
                    "batch_decode did not return valid output"
                )
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
        **kwargs: Any,
    ) -> torch.Tensor:
        from PIL import Image as PILImage

        q = self.to_query(prompt)
        base_text = getattr(q, "text", "") or ""
        full_text = base_text + target
        content: list[dict[str, Any]] = []

        for im in getattr(q, "images", []) or []:
            if isinstance(im, np.ndarray):
                im = PILImage.fromarray(im)
            content.append({"type": "image", "image": im})
        for au in getattr(q, "audios", []) or []:
            content.append({"type": "audio", "audio": au})
        for vid in getattr(q, "videos", []) or []:
            content.append({"type": "video", "video": vid})
        if getattr(q, "text", None):
            content.append({"type": "text", "text": q.text})

        content.append({"type": "text", "text": full_text})
        messages = [{"role": "user", "content": content}]
        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        model_device = next(self.model.parameters()).device
        inputs = {
            k: v.to(model_device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

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
            outputs = self.model(**inputs)
        if not hasattr(outputs, "logits") or outputs.logits is None:
            raise GeneratorError(
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
    def model(self) -> "FastModel":
        if self._model is None:
            self._model, self._processor = self._load_model_from_unsloth()
        return self._model

    @property
    def tokenizer(self) -> Any:
        if hasattr(self._processor, "tokenizer"):
            return self._processor.tokenizer
        if callable(getattr(self._processor, "encode", None)):
            return self._processor
        raise AttributeError(
            f"{self.__class__.__name__}: This processor does not have a `.tokenizer` attribute. "
            "For some multimodal models, please use `.processor` directly."
        )

    @property
    def processor(self) -> Any:
        return self._processor

    @property
    def prompt_template(self) -> str:
        return self._prompt_template

    @prompt_template.setter
    def prompt_template(self, value: str) -> None:
        self._prompt_template = value
