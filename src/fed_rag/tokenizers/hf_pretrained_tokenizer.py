"""HuggingFace PretrainedTokenizer"""

from typing import Any

from pydantic import ConfigDict, Field, PrivateAttr
from transformers import AutoTokenizer, PreTrainedTokenizer

from fed_rag.base.tokenizer import BaseTokenizer


class HFPretrainedTokenizer(BaseTokenizer):
    model_config = ConfigDict(protected_namespaces=("pydantic_model_",))
    model_name: str = Field(
        description="Name of HuggingFace model. Used for loading the model from HF hub or local."
    )
    load_model_kwargs: dict = Field(
        description="Optional kwargs dict for loading models from HF. Defaults to None.",
        default_factory=dict,
    )
    _tokenizer: PreTrainedTokenizer | None = PrivateAttr(default=None)

    def __init__(
        self,
        model_name: str,
        load_model_kwargs: dict | None = None,
        load_model_at_init: bool = True,
    ):
        super().__init__(
            model_name=model_name,
            load_model_kwargs=load_model_kwargs if load_model_kwargs else {},
        )
        if load_model_at_init:
            self._tokenizer = self._load_model_from_hf()

    def _load_model_from_hf(self, **kwargs: Any) -> PreTrainedTokenizer:
        load_kwargs = self.load_model_kwargs
        load_kwargs.update(kwargs)
        self.load_model_kwargs = load_kwargs
        return AutoTokenizer.from_pretrained(self.model_name)

    @property
    def unwrapped_tokenizer(self) -> PreTrainedTokenizer:
        if self._tokenizer is None:
            # load HF Pretrained Tokenizer
            tokenizer = self._load_model_from_hf()
        self._tokenizer = tokenizer
        return self._tokenizer

    @unwrapped_tokenizer.setter
    def unwrapped_tokenizer(self, value: PreTrainedTokenizer) -> None:
        self._tokenizer = value

    def encode(self, input: str, **kwargs: Any) -> list[int]:
        return self.unwrapped_tokenizer.encode(text=input, **kwargs)  # type: ignore[no-any-return]

    def decode(self, input_ids: list[int], **kwargs: Any) -> str:
        return self.unwrapped_tokenizer.decode(token_ids=input_ids, **kwargs)  # type: ignore[no-any-return]
