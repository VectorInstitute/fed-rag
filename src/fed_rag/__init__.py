"""The fed-rag library: simplified fine-tuning for RAG systems"""

import importlib
import sys
from typing import Any

from fed_rag._version import VERSION

# Disable the F403 warning for wildcard imports
# ruff: noqa: F403, F401
from .core import *
from .core import __all__ as _core_all
from .data_structures import RAGConfig
from .knowledge_stores import *
from .knowledge_stores import __all__ as _knowledge_stores_all

_LAZY_MODULES = {
    "generators": "fed_rag.generators",
    "retrievers": "fed_rag.retrievers",
    "trainer_managers": "fed_rag.trainer_managers",
    "trainers": "fed_rag.trainers",
}

_LAZY_CLASSES = {
    # generators
    "HFPretrainedModelGenerator": "fed_rag.generators",
    "HFPeftModelGenerator": "fed_rag.generators",
    "UnslothFastModelGenerator": "fed_rag.generators",
    # retrievers
    "HFSentenceTransformerRetriever": "fed_rag.retrievers",
    # trainers
    "HuggingFaceTrainerForLSR": "fed_rag.trainers",
    "HuggingFaceTrainerForRALT": "fed_rag.trainers",
    # trainer managers
    "HuggingFaceRAGTrainerManager": "fed_rag.trainer_managers",
    "PyTorchRAGTrainerManager": "fed_rag.trainer_managers",
}


def __getattr__(name: str) -> Any:
    """Lazy load modules and classes on first access.

    This is the top-level module's `__getattr__`.
    """

    if name in _LAZY_MODULES:
        module = importlib.import_module(_LAZY_MODULES[name])
        globals()[name] = module
        sys.modules[f"fed_rag.{name}"] = module

        return module

    if name in _LAZY_CLASSES:
        module_name = _LAZY_CLASSES[name]
        module = importlib.import_module(module_name)
        class_obj = getattr(module, name)

        # Cache the class in globals for future access
        globals()[name] = class_obj
        return class_obj

    raise AttributeError(
        f"Module 'fed_rag' has no attribute with name: '{name}'."
    )


def __dir__() -> list[str]:
    """Make lazy classes and modules discoverable"""
    default_attrs = list(globals().keys())  # default module dir
    return sorted(set(default_attrs + __all__))


__version__ = VERSION


__all__ = sorted(
    _core_all
    + _knowledge_stores_all
    + ["RAGConfig"]
    # Lazy loaded modules
    + list(_LAZY_MODULES.keys())
    # Lazy loaded classes
    + list(_LAZY_CLASSES.keys())
)
