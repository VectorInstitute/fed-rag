"""LlamaIndex Bridge"""

from fed_rag.base.bridge import BaseBridgeMixin
from fed_rag.bridges.llamaindex._version import __version__


class LlamaIndexBridgeMixin(BaseBridgeMixin):
    """LlamaIndex Bridge."""

    _bridge_version = __version__
    _bridge_extra = "llama-index"
    _framework = "llama-index-core"
    _compatible_versions = ["0.12.35"]
