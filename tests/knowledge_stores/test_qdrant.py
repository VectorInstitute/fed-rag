import re
import sys
from unittest.mock import patch

import pytest

from fed_rag.exceptions import MissingExtraError
from fed_rag.knowledge_stores.qdrant import QdrantKnowledgeStore


def test_init() -> None:
    knowledge_store = QdrantKnowledgeStore()

    assert isinstance(knowledge_store, QdrantKnowledgeStore)


def test_init_raises_error_if_qdrant_extra_is_missing() -> None:
    modules = {"qdrant_client": None}
    module_to_import = "fed_rag.knowledge_stores.qdrant"

    if module_to_import in sys.modules:
        original_module = sys.modules.pop(module_to_import)

    with patch.dict("sys.modules", modules):
        msg = (
            "Qdrant knowledge stores require the qdrant-client to be installed. "
            "To fix please run `pip install fed-rag[qdrant]`."
        )
        with pytest.raises(
            MissingExtraError,
            match=re.escape(msg),
        ):
            from fed_rag.knowledge_stores.qdrant import QdrantKnowledgeStore

            QdrantKnowledgeStore()

    # restore module so to not affect other tests
    sys.modules[module_to_import] = original_module
