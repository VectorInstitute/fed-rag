from fed_rag.types.rag_system import RAGSystem

from .conftest import MockDataCollator


def test_init(mock_rag_system: RAGSystem) -> None:
    collator = MockDataCollator(rag_system=mock_rag_system)

    assert collator.rag_system == mock_rag_system


def test_collate(mock_rag_system: RAGSystem) -> None:
    collator = MockDataCollator(rag_system=mock_rag_system)

    # act
    res = collator(features={"feat": ["mock_input"]})

    assert collator.rag_system == mock_rag_system
    assert res == "collated!"
