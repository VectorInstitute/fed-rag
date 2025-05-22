"""Unsloth Pretrained Tokenizer"""

import importlib.util
from typing import TYPE_CHECKING

from fed_rag.exceptions import MissingExtraError

from .hf_pretrained_tokenizer import HFPretrainedTokenizer

if importlib.util.find_spec("unsloth") is None:
    _has_unsloth = False
else:
    _has_unsloth = True

if TYPE_CHECKING:  # pragma: no cover
    from transformers import PreTrainedTokenizer


class UnslothPretrainedTokenizer(HFPretrainedTokenizer):
    """Unsloth Pretrained Tokenizer.

    NOTE: Unsloth adds a patch on HF tokenizers, so this is a light wrapper.
    """

    def __init__(
        self,
        tokenizer: "PreTrainedTokenizer",
        model_name: str,
    ):
        if not _has_unsloth:
            msg = (
                f"`{self.__class__.__name__}` requires the `unsloth` extra to be installed. "
                "To fix please run `pip install fed-rag[unsloth]`."
            )
            raise MissingExtraError(msg)
        super().__init__(
            model_name=model_name,
            load_model_at_init=False,
        )
        # set the tokenizer manually as with Unsloth we get patched tokenizer along
        # with the patched model i.e., model, tokenizer = FastModel.from_pretrained(...)
        self._tokenizer = tokenizer
