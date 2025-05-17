"""Bridge type definitions for fed-rag."""

from typing import TypedDict


class BridgeMetadata(TypedDict):
    """Type definition for bridge metadata."""

    bridge_version: str
    framework: str
    compatible_versions: list[str]
    method_name: str
