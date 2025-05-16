"""LlamaIndex Bridge"""

from typing import TYPE_CHECKING

from fed_rag._bridges.llamaindex._version import __version__
from fed_rag.base.bridge import BaseBridgeMixin

if TYPE_CHECKING:  # pragma: no cover
    from llama_index.core.indices.managed.base import BaseManagedIndex

    from fed_rag.core._rag_system import _RAGSystem  # avoids circular import


class LlamaIndexBridgeMixin(BaseBridgeMixin):
    """LlamaIndex Bridge.

    This mixin adds LlamaIndex conversion capabilities to _RAGSystem.
    When mixed with an unbridged _RAGSystem, it allows direct conversion to
    LlamaIndex's BaseManagedIndex through the to_llamaindex() method.
    """

    _bridge_version = __version__
    _bridge_extra = "llama-index"
    _framework = "llama-index"
    _compatible_versions = ["0.12.35"]
    _method_name = "to_llamaindex"

    def to_llamaindex(self: "_RAGSystem") -> "BaseManagedIndex":
        """Converts the _RAGSystem to a ~llamaindex.core.BaseManagedIndex."""
        self._validate_framework_installed()

        from fed_rag._bridges.llamaindex._managed_index import (
            FedRAGManagedIndex,
        )

        return FedRAGManagedIndex(rag_system=self)
