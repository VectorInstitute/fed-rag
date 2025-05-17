"""Knowledge Node

Note: The KnowledgeNOde implementation has moved to fed_rag.data_structures.knowledge_node.
This module is maintained for backward compatibility.
"""

import warnings

from ..data_structures.knowledge_node import (
    KnowledgeNode,
    NodeContent,
    NodeType,
)

warnings.warn(
    "Importing KnowledgeNode, NodeContent, and NodeType from fed_rag.types.knowledge_node"
    "is deprecated and will be removed in a future release. Use "
    "fed_rag.data_structures.knowledge_node or fed_rag.data_structures instead.",
    DeprecationWarning,
    stacklevel=2,  # point to users import statement
)

__all__ = ["KnowledgeNode", "NodeContent", "NodeType"]
