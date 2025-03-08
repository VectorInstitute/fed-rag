from unittest.mock import MagicMock, patch

import pytest

from fed_rag.types.knowledge_node import KnowledgeNode


@patch("fed_rag.types.knowledge_node.uuid")
def test_text_knowledge_node_init(mock_uuid: MagicMock) -> None:
    mock_uuid.uuid4.return_value = "mock_id"
    node = KnowledgeNode(
        embedding=[0.1, 0.2, 0.3], node_type="text", text_content="mock_text"
    )

    assert node.node_id == "mock_id"
    assert node.embedding == [0.1, 0.2, 0.3]
    assert node.node_type == "text"
    assert node.text_content == "mock_text"


def test_text_knowledge_node_init_raises_validation_error() -> None:
    with pytest.raises(
        ValueError, match="NodeType == 'text', but text_content is None."
    ):
        KnowledgeNode(
            node_id="mock_id", embedding=[0.1, 0.2, 0.3], node_type="text"
        )


@patch("fed_rag.types.knowledge_node.uuid")
def test_image_knowledge_node_init(mock_uuid: MagicMock) -> None:
    mock_uuid.uuid4.return_value = "mock_id"
    node = KnowledgeNode(
        embedding=[0.1, 0.2, 0.3],
        node_type="image",
        image_content=b"mock_base64_str",
    )

    assert node.node_id == "mock_id"
    assert node.embedding == [0.1, 0.2, 0.3]
    assert node.node_type == "image"
    assert isinstance(node.image_content, bytes)
    assert node.image_content == b"mock_base64_str"


def test_image_knowledge_node_init_raises_validation_error() -> None:
    with pytest.raises(
        ValueError, match="NodeType == 'image', but image_content is None."
    ):
        KnowledgeNode(
            node_id="mock_id", embedding=[0.1, 0.2, 0.3], node_type="image"
        )
