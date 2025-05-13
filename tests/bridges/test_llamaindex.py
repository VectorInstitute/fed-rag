from typing import ClassVar
from unittest.mock import MagicMock, patch

import pytest
from llama_index.core.schema import MediaResource
from llama_index.core.schema import Node as LlamaNode
from pydantic import BaseModel

from fed_rag.base.bridge import BridgeMetadata
from fed_rag.bridges.llamaindex._managed_index import (
    convert_llama_index_node_to_knowledge_node,
)
from fed_rag.bridges.llamaindex.bridge import LlamaIndexBridgeMixin
from fed_rag.exceptions import BridgeError


# overwrite RAGSystem for this test
class RAGSystem(LlamaIndexBridgeMixin, BaseModel):
    bridges: ClassVar[dict[str, BridgeMetadata]] = {}

    @classmethod
    def _register_bridge(cls, metadata: BridgeMetadata) -> None:
        """To be used only by `BaseBridgeMixin`."""
        if metadata["framework"] not in cls.bridges:
            cls.bridges[metadata["framework"]] = metadata


def test_rag_system_bridges() -> None:
    metadata = LlamaIndexBridgeMixin.get_bridge_metadata()
    rag_system = RAGSystem()

    assert "llama-index" in rag_system.bridges
    assert rag_system.bridges[metadata["framework"]] == metadata
    assert LlamaIndexBridgeMixin._bridge_extra == "llama-index"


@patch("fed_rag.bridges.llamaindex._managed_index.FedRAGManagedIndex")
def test_rag_system_conversion_method(
    mock_managed_index_class: MagicMock,
) -> None:
    metadata = LlamaIndexBridgeMixin.get_bridge_metadata()
    rag_system = RAGSystem()

    conversion_method = getattr(rag_system, metadata["method_name"])
    conversion_method()

    mock_managed_index_class.assert_called_once_with(rag_system=rag_system)


# test node converters
def test_convert_llama_node_to_knowledge_node() -> None:
    llama_node = LlamaNode(
        id_="1",
        embedding=[1, 1, 1],
        text_resource=MediaResource(text="mock text"),
    )
    fed_rag_node = convert_llama_index_node_to_knowledge_node(llama_node)

    assert fed_rag_node.node_id == "1"
    assert fed_rag_node.embedding == [1, 1, 1]
    assert fed_rag_node.node_type == "text"
    assert fed_rag_node.text_content == "mock text"


def test_convert_llama_node_to_knowledge_node_raises_error_missing_embedding() -> (
    None
):
    llama_node = LlamaNode(
        id_="1", text_resource=MediaResource(text="mock text")
    )

    with pytest.raises(
        BridgeError,
        match="Failed to convert ~llama_index.Node: embedding attribute is None.",
    ):
        convert_llama_index_node_to_knowledge_node(llama_node)


def test_convert_llama_node_to_knowledge_node_raises_error_text_resource_is_none() -> (
    None
):
    llama_node = LlamaNode(id_="1", embedding=[1, 1, 1])

    with pytest.raises(
        BridgeError,
        match="Failed to convert ~llama_index.Node: text_resource attribute is None.",
    ):
        convert_llama_index_node_to_knowledge_node(llama_node)


# test FedRAGManagedIndex
