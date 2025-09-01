from typing import TypedDict


class CompatibleVersions(TypedDict, total=False):
    """Type definition for compatible versions.

    Defines optional, inclusive version bounds for compatibility checks.

    Attributes:
        min: Minimum compatible version (inclusive).
        max: Maximum compatible version (inclusive).
    """

    min: str
    max: str


class BridgeMetadata(TypedDict):
    """Type definition for bridge metadata.

    Attributes:
        bridge_version: The version of the bridge.
        framework: The framework name.
        compatible_versions: Version bounds for compatibility.
        method_name: The method name associated with the bridge.
    """

    bridge_version: str
    framework: str
    compatible_versions: CompatibleVersions
    method_name: str
