"""HuggingFace PretrainedTokenizer"""

from typing import TYPE_CHECKING, Any, Optional

from pydantic import ConfigDict, Field, PrivateAttr

from fed_rag.base.tokenizer import BaseTokenizer, EncodeResult
from fed_rag.exceptions.common import MissingExtraError

try:
    from transformers import AutoTokenizer, PreTrainedTokenizer

    _has_huggingface = True
except ModuleNotFoundError:
    _has_huggingface = False


if TYPE_CHECKING:  # pragma: no cover
    from transformers import PreTrainedTokenizer


class HFPretrainedTokenizer(BaseTokenizer):
    model_config = ConfigDict(protected_namespaces=("pydantic_model_",))
    model_name: str = Field(
        description="Name of HuggingFace model. Used for loading the model from HF hub or local."
    )
    load_model_kwargs: dict = Field(
        description="Optional kwargs dict for loading models from HF. Defaults to None.",
        default_factory=dict,
    )
    _tokenizer: Optional["PreTrainedTokenizer"] = PrivateAttr(default=None)

    def __init__(
        self,
        model_name: str,
        load_model_kwargs: dict | None = None,
        load_model_at_init: bool = True,
    ):
        if not _has_huggingface:
            msg = (
                f"`{self.__class__.__name__}` requires `huggingface` extra to be installed. "
                "To fix please run `pip install fed-rag[huggingface]`."
            )
            raise MissingExtraError(msg)
        super().__init__(
            model_name=model_name,
            load_model_kwargs=load_model_kwargs if load_model_kwargs else {},
        )
        if load_model_at_init:
            self._tokenizer = self._load_model_from_hf()

    def _load_model_from_hf(self, **kwargs: Any) -> "PreTrainedTokenizer":
        load_kwargs = self.load_model_kwargs
        load_kwargs.update(kwargs)
        self.load_model_kwargs = load_kwargs
        return AutoTokenizer.from_pretrained(self.model_name, **load_kwargs)

    @property
    def unwrapped(self) -> "PreTrainedTokenizer":
        if self._tokenizer is None:
            # load HF Pretrained Tokenizer
            tokenizer = self._load_model_from_hf()
            self._tokenizer = tokenizer
        return self._tokenizer

    @unwrapped.setter
    def unwrapped(self, value: "PreTrainedTokenizer") -> None:
        self._tokenizer = value

    def encode(self, input: str, **kwargs: Any) -> EncodeResult:
        tokenizer_result = self.unwrapped(text=input, **kwargs)
        retval: EncodeResult = {
            "input_ids": tokenizer_result.get("input_ids"),
            "attention_mask": tokenizer_result.get("attention_mask", None),
        }
        return retval

    def decode(self, input_ids: list[int], **kwargs: Any) -> str:
        return self.unwrapped.decode(token_ids=input_ids, **kwargs)  # type: ignore[no-any-return]
