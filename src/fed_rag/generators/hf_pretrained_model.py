"""HuggingFace PretrainedModel Generator"""

from pydantic import ConfigDict, Field, PrivateAttr
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from transformers.generation.utils import GenerationConfig

from fed_rag.base.generator import BaseGenerator


class HFPretrainedModelGenerator(BaseGenerator):
    model_config = ConfigDict(protected_namespaces=("pydantic_model_",))
    model_name: str = Field(
        description="Name of HuggingFace model. Used for loading the model from HF hub or local."
    )
    generation_config: GenerationConfig = Field(
        description="The generation config used for generating with the PreTrainedModel."
    )
    load_model_kwargs: dict | None = Field(
        description="Optional kwargs dict for loading models from HF. Defaults to None.",
        default=None,
    )
    _model: PreTrainedModel | None = PrivateAttr(default=None)
    _tokenizer: PreTrainedTokenizer | None = PrivateAttr(default=None)

    def __init__(
        self,
        model_name: str,
        generation_config: GenerationConfig | None = None,
        load_model_kwargs: dict | None = None,
        load_model_at_init: bool = True,
    ):
        generation_config = (
            generation_config if generation_config else GenerationConfig()
        )
        super().__init__(
            model_name=model_name,
            generation_config=generation_config,
            load_model_kwargs=load_model_kwargs,
        )
        if load_model_at_init:
            load_model_kwargs = load_model_kwargs if load_model_kwargs else {}
            self._model, self._tokenizer = self._load_model_from_hf(
                **load_model_kwargs
            )

    def _load_model_from_hf(
        self,
    ) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        load_kwargs = self.load_model_kwargs if self.load_model_kwargs else {}
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name, **load_kwargs
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return model, tokenizer

    @property
    def model(self) -> PreTrainedModel:
        if self._model is None:
            # load HF Pretrained Model
            model, _ = self._load_model_from_hf()
            self._model = model
        return self._model

    @model.setter
    def model(self, value: PreTrainedModel) -> None:
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
    def generate(self, input: str) -> str:
        raise NotImplementedError
