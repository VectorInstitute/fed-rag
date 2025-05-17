"""Data structures for results

Note: The correct module has moved to fed_rag.data_structures.results. This module is
maintained for backward compatibility.
"""

import warnings

from ..data_structures.results import TestResult, TrainResult

warnings.warn(
    "Importing TrainResult, TestResult from fed_rag.types.results"
    "is deprecated and will be removed in a future release. Use "
    "fed_rag.data_structures.results or fed_rag.data_structures instead.",
    DeprecationWarning,
    stacklevel=2,  # point to users import statement
)

__all__ = ["TrainResult", "TestResult"]
