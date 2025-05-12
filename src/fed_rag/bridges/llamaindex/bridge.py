"""LlamaIndex Bridge"""

from typing import TYPE_CHECKING

from fed_rag.base.bridge import BaseBridgeMixin
from fed_rag.bridges.llamaindex._version import __version__

if TYPE_CHECKING:
    from llama_index.core.indices.managed.base import BaseManagedIndex

    from fed_rag.types.rag_system import RAGSystem  # avoids circular import


class LlamaIndexBridgeMixin(BaseBridgeMixin):
    """LlamaIndex Bridge."""

    _bridge_version = __version__
    _bridge_extra = "llama-index"
    _framework = "llama-index-core"
    _compatible_versions = ["0.12.35"]

    def to_llamaindex(self: "RAGSystem") -> "BaseManagedIndex":
        """Converts the RAGSystem to a ~llamaindex.core.BaseManagedIndex."""
        from fed_rag.bridges.llamaindex._managed_index import (
            FedRAGManagedIndex,
        )

        return FedRAGManagedIndex(rag_system=self)
