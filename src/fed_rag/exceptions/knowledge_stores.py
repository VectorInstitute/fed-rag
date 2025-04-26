"""Exceptions for Knowledge Stores."""

from .core import FedRAGError


class KnowledgeStoreError(FedRAGError):
    """Base knowledge store error for all knowledge-store-related exceptions."""

    pass


class KnowledgeStoreNotFoundError(KnowledgeStoreError, FileNotFoundError):
    pass
