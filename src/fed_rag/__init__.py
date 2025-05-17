"""The fed-rag library: simplified fine-tuning for RAG systems"""

from fed_rag._version import VERSION

# Disable the F403 warning for wildcard imports
# ruff: noqa: F403, F401
from .core import *
from .core import __all__ as _core_all
from .data_structures import RAGConfig
from .generators import *
from .generators import __all__ as _generators_all
from .knowledge_stores import *
from .knowledge_stores import __all__ as _knowledge_stores_all
from .retrievers import *
from .retrievers import __all__ as _retrievers_all
from .trainer_managers import *
from .trainer_managers import __all__ as _trainer_managers_all

__version__ = VERSION


__all__ = sorted(
    _core_all
    + _generators_all
    + _knowledge_stores_all
    + _retrievers_all
    + _trainer_managers_all
    + ["RAGConfig"]
)
