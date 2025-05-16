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
