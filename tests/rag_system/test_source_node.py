from fed_rag.types.knowledge_node import KnowledgeNode
from fed_rag.types.rag_system import SourceNode


def test_getattr_sourcenode_wraps_knowledge_node() -> None:
    # arrange
    knowledge_node = KnowledgeNode(
        embedding=[0.1, 0.2],
        node_type="text",
        text_content="fake text context",
        metadata={"some_field": 12},
    )

    # act
    source_node = SourceNode(score=0.42, node=knowledge_node)

    # assert
    assert source_node.score == 0.42
    assert source_node.text_content == knowledge_node.text_content
    assert source_node.node_type == knowledge_node.node_type
    assert source_node.node_id == knowledge_node.node_id
    assert source_node.metadata == knowledge_node.metadata
