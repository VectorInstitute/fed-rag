"""HuggingFace PeftModel Generator"""

from typing import TYPE_CHECKING, Any, Optional

from pydantic import ConfigDict, Field, PrivateAttr, model_validator

if TYPE_CHECKING:  # pragma: no cover
    from peft import PeftModel
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


class HFPeftModelGenerator(HuggingFaceGeneratorMixin, BaseGenerator):
    """HFPeftModelGenerator Class.

    NOTE: this class supports loading PeftModel's from HF Hub or from local.
    TODO: support loading custom models via a `~peft.Config` and `~peft.get_peft_model`
    """

    model_config = ConfigDict(protected_namespaces=("pydantic_model_",))
    model_name: str = Field(
        description="Name of Peft model. Used for loading model from HF hub or local."
    )
    base_model_name: str = Field(
        description="Name of the frozen HuggingFace base model. Used for loading the model from HF hub or local."
    )
    generation_config: "GenerationConfig" = Field(
        description="The generation config used for generating with the PreTrainedModel."
    )
    load_model_kwargs: dict = Field(
        description="Optional kwargs dict for loading peft model from HF. Defaults to None.",
        default_factory=dict,
    )
    load_base_model_kwargs: dict = Field(
        description="Optional kwargs dict for loading base model from HF. Defaults to None.",
        default_factory=dict,
    )
    _prompt_template: str = PrivateAttr(default=DEFAULT_PROMPT_TEMPLATE)
    _model: Optional["PeftModel"] = PrivateAttr(default=None)
    _tokenizer: HFPretrainedTokenizer | None = PrivateAttr(default=None)

    def __init__(
        self,
        model_name: str,
        base_model_name: str,
        generation_config: Optional["GenerationConfig"] = None,
        prompt_template: str | None = None,
        load_model_kwargs: dict | None = None,
        load_base_model_kwargs: dict | None = None,
        load_model_at_init: bool = True,
    ):
        # if reaches here, then passed checks for huggingface extra installation
        from transformers.generation.utils import GenerationConfig

        generation_config = (
            generation_config if generation_config else GenerationConfig()
        )
        super().__init__(
            model_name=model_name,
            base_model_name=base_model_name,
            generation_config=generation_config,
            prompt_template=prompt_template,
            load_model_kwargs=load_model_kwargs if load_model_kwargs else {},
            load_base_model_kwargs=(
                load_base_model_kwargs if load_base_model_kwargs else {}
            ),
        )
        self._tokenizer = HFPretrainedTokenizer(
            model_name=base_model_name, load_model_at_init=load_model_at_init
        )
        self._prompt_template = (
            prompt_template if prompt_template else DEFAULT_PROMPT_TEMPLATE
        )
        if load_model_at_init:
            self._model = self._load_model_from_hf()

    def _load_model_from_hf(self, **kwargs: Any) -> "PeftModel":
        from peft import PeftModel, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM

        load_base_kwargs = self.load_base_model_kwargs
        load_kwargs = self.load_model_kwargs
        load_kwargs.update(kwargs)
        self.load_model_kwargs = load_kwargs  # update load_model_kwargs
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name, **load_base_kwargs
        )

        if "quantization_config" in load_base_kwargs:
            # preprocess model for kbit fine-tuning
            # https://huggingface.co/docs/peft/developer_guides/quantization
            base_model = prepare_model_for_kbit_training(base_model)

        return PeftModel.from_pretrained(
            base_model, self.model_name, **load_kwargs
        )

    @model_validator(mode="before")
    @classmethod
    def check_dependencies(cls, data: Any) -> Any:
        """Validate that huggingface dependencies are installed."""
        check_huggingface_installed(cls.__name__)
        return data

    @property
    def model(self) -> "PeftModel":
        if self._model is None:
            # load HF PeftModel
            self._model = self._load_model_from_hf()
        return self._model

    @model.setter
    def model(self, value: "PeftModel") -> None:
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
