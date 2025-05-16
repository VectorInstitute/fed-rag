"""Public API for fed-rag (Pre-0.1.0)

This module defines the evolving public API for fed-rag. While we aim for stability,
components here may change during the pre-0.1.0 development phase.

For most use cases, this is the only module you need to import from: fed_rag.api.
"""

# Disable the F403 warning for wildcard imports
# ruff: noqa: F403, F401
from .core import *
from .core import __all__ as _core_all
from .types import *
from .types import __all__ as _types_all

API_VERSION = "0.0.1"


def get_api_version() -> str:
    """Return the current API version."""
    return API_VERSION


__all__ = _core_all + _types_all + ["API_VERSION", "get_api_version"]
