import re
import sys
from unittest.mock import MagicMock, patch

import pytest

from fed_rag.base.tokenizer import BaseTokenizer
from fed_rag.exceptions import MissingExtraError
from fed_rag.tokenizers.unsloth_pretrained_tokenizer import (
    UnslothPretrainedTokenizer,
)


def test_hf_pretrained_generator_class() -> None:
    names_of_base_classes = [
        b.__name__ for b in UnslothPretrainedTokenizer.__mro__
    ]
    assert BaseTokenizer.__name__ in names_of_base_classes


def test_unsloth_extra_missing() -> None:
    """Test extra is not installed."""

    modules = {"unsloth": None}
    module_to_import = "fed_rag.tokenizers.unsloth_pretrained_tokenizer"

    if module_to_import in sys.modules:
        original_module = sys.modules.pop(module_to_import)

    with patch.dict("sys.modules", modules):
        msg = (
            "`UnslothPretrainedTokenizer` requires the `unsloth` extra to be installed. "
            "To fix please run `pip install fed-rag[unsloth]`."
        )
        with pytest.raises(
            MissingExtraError,
            match=re.escape(msg),
        ):
            from fed_rag.tokenizers.unsloth_pretrained_tokenizer import (
                UnslothPretrainedTokenizer,
            )

            mock_tokenizer = MagicMock()

            UnslothPretrainedTokenizer(mock_tokenizer, "fake_name")

    # restore module so to not affect other tests
    sys.modules[module_to_import] = original_module
