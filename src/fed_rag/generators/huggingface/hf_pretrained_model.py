"""HuggingFace PretrainedModel Generator"""

from typing import TYPE_CHECKING, Any, Optional

from pydantic import ConfigDict, Field, PrivateAttr, model_validator

if TYPE_CHECKING:  # pragma: no cover
    from transformers import PreTrainedModel
    from transformers.generation.utils import GenerationConfig

from fed_rag.base.generator import BaseGenerator
from fed_rag.tokenizers.hf_pretrained_tokenizer import HFPretrainedTokenizer

from .mixin import HuggingFaceGeneratorMixin
from .utils import check_huggingface_installed

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


class HFPretrainedModelGenerator(HuggingFaceGeneratorMixin, BaseGenerator):
    model_config = ConfigDict(protected_namespaces=("pydantic_model_",))
    model_name: str = Field(
        description="Name of HuggingFace model. Used for loading the model from HF hub or local."
    )
    generation_config: "GenerationConfig" = Field(
        description="The generation config used for generating with the PreTrainedModel."
    )
    load_model_kwargs: dict = Field(
        description="Optional kwargs dict for loading models from HF. Defaults to None.",
        default_factory=dict,
    )
    _prompt_template: str = PrivateAttr(default=DEFAULT_PROMPT_TEMPLATE)
    _model: Optional["PreTrainedModel"] = PrivateAttr(default=None)
    _tokenizer: HFPretrainedTokenizer | None = PrivateAttr(default=None)

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
        self._tokenizer = HFPretrainedTokenizer(
            model_name=model_name, load_model_at_init=load_model_at_init
        )
        self._prompt_template = (
            prompt_template if prompt_template else DEFAULT_PROMPT_TEMPLATE
        )
        if load_model_at_init:
            self._model = self._load_model_from_hf()

    @model_validator(mode="before")
    @classmethod
    def check_dependencies(cls, data: Any) -> Any:
        """Validate that qdrant dependencies are installed."""
        check_huggingface_installed(cls.__name__)
        return data

    def _load_model_from_hf(self, **kwargs: Any) -> "PreTrainedModel":
        from transformers import AutoModelForCausalLM

        load_kwargs = self.load_model_kwargs
        load_kwargs.update(kwargs)
        self.load_model_kwargs = load_kwargs
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name, **load_kwargs
        )
        return model

    @property
    def model(self) -> "PreTrainedModel":
        if self._model is None:
            # load HF Pretrained Model
            model = self._load_model_from_hf()
            self._model = model
        return self._model

    @model.setter
    def model(self, value: "PreTrainedModel") -> None:
        self._model = value

    @property
    def tokenizer(self) -> HFPretrainedTokenizer:
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, value: HFPretrainedTokenizer) -> None:
        self._tokenizer = value

    @property
    def prompt_template(self) -> str:
        return self._prompt_template

    @prompt_template.setter
    def prompt_template(self, value: str) -> None:
        self._prompt_template = value
