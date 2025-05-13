from typing import ClassVar

from pydantic import BaseModel

from fed_rag.base.bridge import BridgeMetadata
from fed_rag.bridges.llamaindex.bridge import LlamaIndexBridgeMixin


# overwrite RAGSystem for this test
class RAGSystem(LlamaIndexBridgeMixin, BaseModel):
    bridges: ClassVar[dict[str, BridgeMetadata]] = {}

    @classmethod
    def _register_bridge(cls, metadata: BridgeMetadata) -> None:
        """To be used only by `BaseBridgeMixin`."""
        if metadata["framework"] not in cls.bridges:
            cls.bridges[metadata["framework"]] = metadata


def test_rag_system_bridges() -> None:
    metadata = LlamaIndexBridgeMixin.get_bridge_metadata()
    rag_system = RAGSystem()

    assert "llama-index-core" in rag_system.bridges
    assert rag_system.bridges[metadata["framework"]] == metadata
    assert LlamaIndexBridgeMixin._bridge_extra == "llama-index"
