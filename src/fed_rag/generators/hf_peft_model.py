"""HuggingFace PeftModel Generator"""

from typing import Any

import torch
from peft import PeftModel, prepare_model_for_kbit_training
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

        model = PeftModel.from_pretrained(
            base_model, self.model_name, **load_kwargs
        )
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        return model, tokenizer

    @property
    def model(self) -> PeftModel:
        if self._model is None:
            # load HF PeftModel
            model, _ = self._load_model_from_hf()
            self._model = model
        return self._model

    @model.setter
    def model(self, value: PeftModel) -> None:
        self._model = value

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        if self._tokenizer is None:
            # load HF Pretrained Model
            _, tokenizer = self._load_model_from_hf()
            self._tokenizer = tokenizer
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, value: PreTrainedTokenizer) -> None:
        self._tokenizer = value

    # generate
    def generate(self, query: str, context: str, **kwargs: Any) -> str:
        formatted_query = self.prompt_template.format(
            question=query, context=context
        )

        # encode query
        tokenizer_result = self.tokenizer(formatted_query, return_tensors="pt")
        inputs: torch.Tensor = tokenizer_result.input_ids
        inputs = inputs.to(self.model.device)

        # generate
        generated_ids = self.model.generate(
            inputs=inputs,
            generation_config=self.generation_config,
            tokenizer=self._tokenizer,
            **kwargs,
        )

        # decode tokens
        outputs: list[str] = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        return outputs[0]
