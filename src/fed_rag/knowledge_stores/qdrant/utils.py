"""Qdrant utils module."""

from importlib.util import find_spec

from fed_rag.exceptions import MissingExtraError


def check_qdrant_installed() -> None:
    if find_spec("qdrant_client") is None:
        raise MissingExtraError(
            "Qdrant knowledge stores require the qdrant-client to be installed. "
            "To fix please run `pip install fed-rag[qdrant]`."
        )
