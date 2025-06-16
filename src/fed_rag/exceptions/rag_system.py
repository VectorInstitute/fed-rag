"""Exceptions for RAG System."""

from .core import FedRAGError, FedRAGWarning


class RAGSystemError(FedRAGError):
    """Base evals error for all generator-related exceptions."""

    pass


class RAGSystemWarning(FedRAGWarning):
    """Base inspector warning for all generator-related warnings."""

    pass
