"""Bridge type definitions for fed-rag."""

from typing import TypedDict


class CompatibleVersions(TypedDict, total=False):
    """Type definition for compatible versions.

    Defines optional, inclusive version bounds for compatibility checks.
    """

    min: str
    max: str


class BridgeMetadata(TypedDict):
    """Type definition for bridge metadata."""

    bridge_version: str
    framework: str
    compatible_versions: CompatibleVersions
    method_name: str
