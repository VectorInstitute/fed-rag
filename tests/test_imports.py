import importlib
from contextlib import nullcontext as does_not_raise

import pytest

from fed_rag import types


def test_root_imports() -> None:
    """Test that core types can be imported from the root."""
    with does_not_raise():
        # ruff: noqa: F401
        from fed_rag import RAGConfig, RAGSystem


def test_type_imports() -> None:
    """Test that there are no circular imports in the types module."""
    with does_not_raise():
        # ruff: noqa: F401
        from fed_rag.types import (  # RAGConfig,; RAGSystem,
            KnowledgeNode,
            NodeContent,
            NodeType,
            RAGResponse,
            SourceNode,
            TestResult,
            TrainResult,
        )


def test_base_direct_imports() -> None:
    """Test that base classes can be imported directly."""
    with does_not_raise():
        # ruff: noqa: F401
        from fed_rag.base.bridge import BaseBridgeMixin
        from fed_rag.base.data_collator import BaseDataCollator
        from fed_rag.base.fl_task import BaseFLTask
        from fed_rag.base.generator import BaseGenerator
        from fed_rag.base.knowledge_store import BaseKnowledgeStore
        from fed_rag.base.retriever import BaseRetriever
        from fed_rag.base.tokenizer import BaseTokenizer
        from fed_rag.base.trainer import (
            BaseGeneratorTrainer,
            BaseRetrieverTrainer,
            BaseTrainer,
        )
        from fed_rag.base.trainer_config import BaseTrainerConfig
        from fed_rag.base.trainer_manager import BaseRAGTrainerManager


def test_all_simultaneously() -> None:
    """Test that there are no circular imports."""
    with does_not_raise():
        # ruff: noqa: F401
        from fed_rag import RAGConfig, RAGSystem
        from fed_rag.base.bridge import BaseBridgeMixin
        from fed_rag.base.data_collator import BaseDataCollator
        from fed_rag.base.fl_task import BaseFLTask
        from fed_rag.base.generator import BaseGenerator
        from fed_rag.base.knowledge_store import BaseKnowledgeStore
        from fed_rag.base.retriever import BaseRetriever
        from fed_rag.base.tokenizer import BaseTokenizer
        from fed_rag.base.trainer import (
            BaseGeneratorTrainer,
            BaseRetrieverTrainer,
            BaseTrainer,
        )
        from fed_rag.base.trainer_config import BaseTrainerConfig
        from fed_rag.base.trainer_manager import BaseRAGTrainerManager
        from fed_rag.types import (
            KnowledgeNode,
            NodeContent,
            NodeType,
            RAGResponse,
            SourceNode,
            TestResult,
            TrainResult,
        )


@pytest.mark.parametrize("name", types.__all__)
def test_types_all_importable(name: str) -> None:
    """Tests that all names listed in __all__ are importable."""
    mod = importlib.import_module("fed_rag.types")
    attr = getattr(mod, name)

    assert hasattr(mod, name)
    assert attr is not None


def test_import_rag_system_from_types() -> None:
    """Test deprecated import path for RAGSystem from fed_rag.types"""

    with pytest.warns(DeprecationWarning):
        # ruff: noqa: F401
        from fed_rag.types.rag_system import RAGSystem
