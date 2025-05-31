"""LangChain Bridge"""

from typing import TYPE_CHECKING

from fed_rag._bridges.langchain._version import __version__
from fed_rag.base.bridge import BaseBridgeMixin

if TYPE_CHECKING:  # pragma: no cover
    from langchain_core.language_models import BaseLLM
    from langchain_core.vectorstores import VectorStore

    from fed_rag.core.rag_system._synchronous import (
        _RAGSystem,  # avoids circular import
    )


class LangChainBridgeMixin(BaseBridgeMixin):
    """LangChain Bridge.

    This mixin adds LangChain conversion capabilities to _RAGSystem.
    When mixed with an unbridged _RAGSystem, it allows direct conversion to
    LangChain's VectorStore and BaseLLM through the to_langchain() method.
    """

    _bridge_version = __version__
    _bridge_extra = "langchain"
    _framework = "langchain-core"
    _compatible_versions = {"min": "0.3.62"}
    _method_name = "to_langchain"

    def to_langchain(self: "_RAGSystem") -> tuple["VectorStore", "BaseLLM"]:
        """Converts the _RAGSystem to a tuple of ~langchain_core.vectorstores.VectorStore and ~langchain_core.language_models.BaseLLM."""
        self._validate_framework_installed()

        from fed_rag._bridges.langchain._bridge_classes import (
            FedRAGLLM,
            FedRAGVectorStore,
        )

        return FedRAGVectorStore(self), FedRAGLLM(self)
