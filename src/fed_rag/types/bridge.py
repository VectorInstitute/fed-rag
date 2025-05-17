"""Bridge type definitions for fed-rag.

Note: The BridgeMetadata implementation has moved to fed_rag.data_structures.bridge.
This module is maintained for backward compatibility.
"""

import warnings

from ..data_structures.bridge import BridgeMetadata

warnings.warn(
    "Importing BridgeMetadata from fed_rag.types.bridge is deprecated and will be "
    "removed in a future release. Use fed_rag.data_structures.bridge or "
    "fed_rag.data_structures instead.",
    DeprecationWarning,
    stacklevel=2,  # point to users import statement
)

__all__ = ["BridgeMetadata"]
