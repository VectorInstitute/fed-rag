"""HuggingFace PeftModel Generator"""

from typing import Any

from peft import PeftModel
from pydantic import ConfigDict, Field, PrivateAttr
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
)
from transformers.generation.utils import GenerationConfig

from fed_rag.base.generator import BaseGenerator

DEFAULT_PROMPT_TEMPLATE = """
You are a helpful assistant. Given the user's question, provide a succinct
and accurate response. If context is provided, use it in your answer if it helps
you to create the most accurate response.

<question>
{question}
</question>

<context>
{context}
</context>

<response>

"""


class HFPeftModelGenerator(BaseGenerator):
    model_config = ConfigDict(protected_namespaces=("pydantic_model_",))
    model_name: str = Field(
        description="Name of Peft model. Used for loading model from HF hub or local."
    )
    base_model_name: str = Field(
        description="Name of the frozen HuggingFace base model. Used for loading the model from HF hub or local."
    )
    generation_config: GenerationConfig = Field(
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
    prompt_template: str = Field(description="Prompt template for RAG.")
    _model: PeftModel | None = PrivateAttr(default=None)
    _tokenizer: PreTrainedTokenizer | None = PrivateAttr(default=None)

    def __init__(
        self,
        model_name: str,
        base_model_name: str,
        generation_config: GenerationConfig | None = None,
        prompt_template: str | None = None,
        load_model_kwargs: dict | None = None,
        load_base_model_kwargs: dict | None = None,
        load_model_at_init: bool = True,
    ):
        generation_config = (
            generation_config if generation_config else GenerationConfig()
        )
        prompt_template = (
            prompt_template if prompt_template else DEFAULT_PROMPT_TEMPLATE
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
        if load_model_at_init:
            self._model, self._tokenizer = self._load_model_from_hf()

    def _load_model_from_hf(
        self, **kwargs: Any
    ) -> tuple[PeftModel, PreTrainedTokenizer]:
        load_kwargs = self.load_model_kwargs
        load_kwargs.update(kwargs)
        self.load_model_kwargs = load_kwargs
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name, **load_kwargs
        )
        model = PeftModel.from_pretrained(
            base_model, self.model_name, **load_kwargs
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return model, tokenizer

    @property
    def model(self) -> PeftModel:
        if self._model is None:
            # load HF PeftModel
            model, _ = self._load_model_from_hf(**self.load_model_kwargs)
            self._model = model
        return self._model

    @model.setter
    def model(self, value: PeftModel) -> None:
        self._model = value
