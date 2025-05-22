"""Unsloth FastModel Generator"""

from typing import TYPE_CHECKING, Any, Optional

from pydantic import ConfigDict, Field, PrivateAttr, model_validator

if TYPE_CHECKING:  # pragma: no cover
    from unsloth import FastLanguageModel
    from transformers import PreTrainedModel, PreTrainedTokenizer
    from transformers.generation.utils import GenerationConfig

from fed_rag.base.generator import BaseGenerator
from fed_rag.tokenizers.unsloth_pretrained_tokenizer import (
    UnslothPretrainedTokenizer,
)

from .mixin import UnslothGeneratorMixin
from .utils import check_unsloth_installed

DEFAULT_PROMPT_TEMPLATE = """
You are a helpful assistant. Given the user's query, provide a succinct
and accurate response. If context is provided, use it in your answer if it helps
you to create the most accurate response.

<query>
{query}
</query>

<context>
{context}
</context>

<response>

"""


class UnslothFastModelGenerator(UnslothGeneratorMixin, BaseGenerator):
    model_config = ConfigDict(protected_namespaces=("pydantic_model_",))
    model_name: str = Field(
        description="Name of Unsloth model. Used for loading the model from HF hub or local."
    )
    generation_config: "GenerationConfig" = Field(
        description="The generation config used for generating with the PreTrainedModel."
    )
    load_model_kwargs: dict = Field(
        description="Optional kwargs dict for loading ~unsloth.FastModel.from_pretrained(). Defaults to None.",
        default_factory=dict,
    )
    _prompt_template: str = PrivateAttr(default=DEFAULT_PROMPT_TEMPLATE)
    _model: Optional["FastLanguageModel"] = PrivateAttr(default=None)
    _tokenizer: UnslothPretrainedTokenizer | None = PrivateAttr(default=None)

    def __init__(
        self,
        model_name: str,
        generation_config: Optional["GenerationConfig"] = None,
        prompt_template: str | None = None,
        load_model_kwargs: dict | None = None,
        load_model_at_init: bool = True,
    ):
        # if reaches here, then passed checks for extra
        from transformers.generation.utils import GenerationConfig

        generation_config = (
            generation_config if generation_config else GenerationConfig()
        )
        super().__init__(
            model_name=model_name,
            generation_config=generation_config,
            load_model_kwargs=load_model_kwargs if load_model_kwargs else {},
        )
        self._prompt_template = (
            prompt_template if prompt_template else DEFAULT_PROMPT_TEMPLATE
        )
        if load_model_at_init:
            self._model, tokenizer = self._load_model_and_tokenizer()
            self._tokenizer = UnslothPretrainedTokenizer(
                model_name=self.model_name, tokenizer=tokenizer
            )

    @model_validator(mode="before")
    @classmethod
    def check_dependencies(cls, data: Any) -> Any:
        """Validate that qdrant dependencies are installed."""
        check_unsloth_installed(cls.__name__)
        return data

    def _load_model_and_tokenizer(
        self, **kwargs: Any
    ) -> tuple["PreTrainedModel", "PreTrainedTokenizer"]:
        from unsloth import FastLanguageModel

        load_kwargs = self.load_model_kwargs
        load_kwargs.update(kwargs)
        self.load_model_kwargs = load_kwargs
        model, tokenizer = FastLanguageModel.from_pretrained(
            self.model_name, **load_kwargs
        )
        return model, tokenizer

    @property
    def model(self) -> "PreTrainedModel":
        if self._model is None:
            # load HF Pretrained Model
            model, tokenizer = self._load_model_and_tokenizer()
            self._model = model
            if self._tokenizer is None:
                self._tokenizer = UnslothPretrainedTokenizer(
                    model_name=self.model_name, tokenizer=tokenizer
                )
        return self._model

    @model.setter
    def model(self, value: "PreTrainedModel") -> None:
        self._model = value

    @property
    def tokenizer(self) -> UnslothPretrainedTokenizer:
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, value: UnslothPretrainedTokenizer) -> None:
        self._tokenizer = value

    @property
    def prompt_template(self) -> str:
        return self._prompt_template

    @prompt_template.setter
    def prompt_template(self, value: str) -> None:
        self._prompt_template = value
