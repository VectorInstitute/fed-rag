"""Exceptions for Retrievers."""

from .core import FedRAGError, FedRAGWarning


class RetrieverError(FedRAGError):
    """Base evals error for all retriever-related exceptions."""

    pass


class RetrieverWarning(FedRAGWarning):
    """Base inspector warning for all retriever-related warnings."""

    pass
