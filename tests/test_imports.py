from contextlib import nullcontext as does_not_raise


def test_type_imports() -> None:
    """Test that there are no circular imports in the types module."""
    with does_not_raise():
        # ruff: noqa: F401
        from fed_rag.types import (
            KnowledgeNode,
            NodeContent,
            NodeType,
            RAGConfig,
            RAGResponse,
            RAGSystem,
            SourceNode,
            TestResult,
            TrainResult,
        )


def test_root_imports() -> None:
    """Test that core types can be imported from the root."""
    with does_not_raise():
        # ruff: noqa: F401
        from fed_rag import RAGConfig, RAGSystem


def test_base_direct_imports() -> None:
    """Test that base classes can be imported directly."""
    with does_not_raise():
        # ruff: noqa: F401
        from fed_rag.base.bridge import BaseBridgeMixin
        from fed_rag.base.data_collator import BaseDataCollator
