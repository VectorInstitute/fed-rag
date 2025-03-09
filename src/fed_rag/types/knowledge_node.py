"""Knowledge Node"""

import uuid
from enum import Enum
from typing import TypedDict, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
)


class NodeContent(TypedDict):
    text_content: str | None
    image_content: bytes | None


class NodeType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    MULTIMODAL = "multimodal"


class KnowledgeNode(BaseModel):
    model_config = ConfigDict(
        # ensures that validation is performed for defaulted None values
        validate_default=True
    )
    node_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    embedding: list[float] = Field(
        description="Encoded representation of node. If multimodal type, then this is shared embedding between image and text."
    )
    node_type: NodeType = Field(description="Type of node.")
    text_content: str | None = Field(
        description="Text content. Used for TEXT and potentially MULTIMODAL node types.",
        default=None,
    )
    image_content: bytes | None = Field(
        description="Image content as binary data (decoded from base64)",
        default=None,
    )
    metadata: dict = Field(
        description="Metadata for node.", default_factory=dict
    )

    # validators
    @field_validator("text_content", mode="before")
    @classmethod
    def validate_text_content(
        cls, value: str | None, info: ValidationInfo
    ) -> str | None:
        node_type = info.data.get("node_type")
        node_type = cast(NodeType, node_type)
        if node_type == NodeType.TEXT and value is None:
            raise ValueError("NodeType == 'text', but text_content is None.")

        if node_type == NodeType.MULTIMODAL and value is None:
            raise ValueError(
                "NodeType == 'multimodal', but text_content is None."
            )

        return value

    @field_validator("image_content", mode="after")
    @classmethod
    def validate_image_content(
        cls, value: str | None, info: ValidationInfo
    ) -> str | None:
        node_type = info.data.get("node_type")
        node_type = cast(NodeType, node_type)
        if node_type == NodeType.IMAGE:
            if value is None:
                raise ValueError(
                    "NodeType == 'image', but image_content is None."
                )

        if node_type == NodeType.MULTIMODAL:
            if value is None:
                raise ValueError(
                    "NodeType == 'multimodal', but image_content is None."
                )

        return value

    def get_content(self) -> NodeContent:
        """Return dict of node content."""
        content: NodeContent = {
            "image_content": self.image_content,
            "text_content": self.text_content,
        }
        return content
