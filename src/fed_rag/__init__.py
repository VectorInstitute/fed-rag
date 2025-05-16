import importlib
from importlib.metadata import version
from types import ModuleType

# Expose only the main user-facing class at root level
from fed_rag.types.rag_system import RAGConfig, RAGSystem

try:
    __version__ = version("fed-rag")
except ImportError:
    __version__ = "unknown"  # fallback for development installs


# Set up lazy loading for submodules
def __getattr__(name: str) -> ModuleType:
    """Lazy-load submodules on demand."""
    if name in _SUBMODULES:
        module = importlib.import_module(f"fed_rag.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module 'fed_rag' has no attribute '{name}'")


# Define submodules for lazy loading
_SUBMODULES = [
    "_bridges",
    "base",
    "data_collators",
    "decorators",
    "exceptions",
    "fl_tasks",
    "generators",
    "inspectors",
    "knowledge_stores",
    "loss",
    "retrievers",
    "tokenizers",
    "trainer_configs",
    "trainer_managers",
    "trainers",
    "types",
    "utils",
]

__all__ = ["RAGSystem", "RAGConfig"]
