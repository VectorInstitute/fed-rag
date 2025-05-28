"""Common exceptions."""

from .core import FedRAGError


class MissingExtraError(FedRAGError):
    """Raised when a fed-rag extra is not installed."""
