import sys

import pytest

DEPRECATED_IMPORTS = [
    ("fed_rag.types.bridge", "BridgeMetadata"),
    ("fed_rag.types.results", "TrainResult"),
    ("fed_rag.types.results", "TestResult"),
    ("fed_rag.types.knowledge_node", "KnowledgeNode"),
    ("fed_rag.types.knowledge_node", "NodeType"),
    ("fed_rag.types.knowledge_node", "NodeContent"),
    ("fed_rag.types.rag", "RAGConfig"),
    ("fed_rag.types.rag", "RAGResponse"),
    ("fed_rag.types.rag", "SourceNode"),
]


@pytest.mark.parametrize("module_path,class_name", DEPRECATED_IMPORTS)
def test_import_from_types_raises_deprecation_warning(
    module_path: str, class_name: str
) -> None:
    """Test that importing from deprecated types modules raises warnings."""

    # clear the module from sys.modules if it exists
    if module_path in sys.modules:
        del sys.modules[module_path]

    with pytest.warns(DeprecationWarning):
        import importlib

        module = importlib.import_module(module_path)
        getattr(module, class_name)  # ensure its loaded
