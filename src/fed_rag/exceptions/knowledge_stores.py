"""Exceptions for Knowledge Stores."""


class KnowledgeStoreError(Exception):
    """Base knowledge store error for all knowledge-store-related exceptions."""

    pass


class KnowledgeStoreNotFoundError(KnowledgeStoreError, FileNotFoundError):
    pass
