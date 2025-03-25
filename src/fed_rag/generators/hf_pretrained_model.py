"""HuggingFace PretrainedModel Generator"""

from typing import Any

import torch
from pydantic import ConfigDict, Field, PrivateAttr
from transformers import AutoModelForCausalLM, PreTrainedModel
from transformers.generation.utils import GenerationConfig

from fed_rag.base.generator import BaseGenerator
from fed_rag.tokenizers.hf_pretrained_tokenizer import HFPretrainedTokenizer

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


class HFPretrainedModelGenerator(BaseGenerator):
    model_config = ConfigDict(protected_namespaces=("pydantic_model_",))
    model_name: str = Field(
        description="Name of HuggingFace model. Used for loading the model from HF hub or local."
    )
    generation_config: GenerationConfig = Field(
        description="The generation config used for generating with the PreTrainedModel."
    )
    load_model_kwargs: dict = Field(
        description="Optional kwargs dict for loading models from HF. Defaults to None.",
        default_factory=dict,
    )
    prompt_template: str = Field(description="Prompt template for RAG.")
    _model: PreTrainedModel | None = PrivateAttr(default=None)
    _tokenizer: HFPretrainedTokenizer | None = PrivateAttr(default=None)

    def __init__(
        self,
        model_name: str,
        generation_config: GenerationConfig | None = None,
        prompt_template: str | None = None,
        load_model_kwargs: dict | None = None,
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
            generation_config=generation_config,
            prompt_template=prompt_template,
            load_model_kwargs=load_model_kwargs if load_model_kwargs else {},
        )
        self._tokenizer = HFPretrainedTokenizer(
            model_name=model_name, load_model_at_init=load_model_at_init
        )
        if load_model_at_init:
            self._model = self._load_model_from_hf()

    def _load_model_from_hf(self, **kwargs: Any) -> PreTrainedModel:
        load_kwargs = self.load_model_kwargs
        load_kwargs.update(kwargs)
        self.load_model_kwargs = load_kwargs
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name, **load_kwargs
        )
        return model

    @property
    def model(self) -> PreTrainedModel:
        if self._model is None:
            # load HF Pretrained Model
            model = self._load_model_from_hf()
            self._model = model
        return self._model

    @model.setter
    def model(self, value: PreTrainedModel) -> None:
        self._model = value

    @property
    def tokenizer(self) -> HFPretrainedTokenizer:
        return self._tokenizer

    # generate
    def generate(self, query: str, context: str, **kwargs: Any) -> str:
        formatted_query = self.prompt_template.format(
            question=query, context=context
        )

        # encode query
        tokenizer_result = self.tokenizer.unwrapped(
            formatted_query, return_tensors="pt"
        )
        inputs: torch.Tensor = tokenizer_result.input_ids
        inputs = inputs.to(self.model.device)

        # generate
        generated_ids = self.model.generate(
            inputs=inputs,
            generation_config=self.generation_config,
            tokenizer=self.tokenizer.unwrapped,
            **kwargs,
        )

        # decode tokens
        outputs: list[str] = self.tokenizer.unwrapped.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        return outputs[0]
