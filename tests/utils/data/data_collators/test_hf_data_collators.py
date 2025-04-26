import re
import sys
from unittest.mock import patch

import pytest

from fed_rag.exceptions import MissingExtraError
from fed_rag.exceptions.core import FedRAGError
from fed_rag.types.rag_system import RAGSystem


def test_huggingface_extra_missing(mock_rag_system: RAGSystem) -> None:
    """Test extra is not installed."""

    modules = {
        "transformers": None,
        "transformers.data": None,
        "transformers.data.data_collator": None,
    }
    module_to_import = "fed_rag.utils.data.data_collators.huggingface"
    original_module = sys.modules.pop(module_to_import, None)

    with patch.dict("sys.modules", modules):
        msg = (
            "`DataCollatorForLSR` requires `huggingface` extra to be installed. "
            "To fix please run `pip install fed-rag[huggingface]`."
        )
        with pytest.raises(
            MissingExtraError,
            match=re.escape(msg),
        ):
            from fed_rag.utils.data.data_collators.huggingface import (
                DataCollatorForLSR,
            )

            DataCollatorForLSR(rag_system=mock_rag_system, prompt_template="")

    # restore module so to not affect other tests
    if original_module:
        sys.modules[module_to_import] = original_module


def test_invalid_rag_system_due_to_generators(
    mock_rag_system: RAGSystem,
) -> None:
    """Test extra is not installed."""

    with pytest.raises(
        FedRAGError,
        match="Generator must be HFPretrainedModelGenerator or HFPeftModelGenerator",
    ):
        from fed_rag.utils.data.data_collators.huggingface import (
            DataCollatorForLSR,
        )

        DataCollatorForLSR(rag_system=mock_rag_system, prompt_template="")
