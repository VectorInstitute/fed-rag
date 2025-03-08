"""Knowledge Node"""

import uuid
from enum import Enum

from pydantic import BaseModel, Field, ValidationInfo, field_validator


class NodeType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    MULTIMODAL = "multimodal"


class KnowledgeNode(BaseModel):
    node_id = Field(default_factory=lambda: str(uuid.uuid4()))
    embedding: list[float] = Field(
        description="Encoded representation of node."
    )
    node_type: NodeType = Field(description="Type of node.")
    text_content: str | None = Field(
        description="Text content. Used for TEXT and potentially MULTIMODAL node types."
    )
    image_content: bytes | None = Field(
        description="Image content as binary data (decoded from base64)"
    )
    metadata: dict = Field(
        description="Metadata for node.", default_factory=dict
    )

    def get_content(self) -> str | bytes | dict | None:
        """Return the appropriate content based on node_type."""
        if self.node_type == NodeType.TEXT:
            return self.text_content
        elif self.node_type == NodeType.IMAGE:
            return self.image_content
        elif self.node_type == NodeType.MULTIMODAL:
            return {"text": self.text_content, "image": self.image_content}

    # validators
    @field_validator("text_content", mode="after")
    @classmethod
    def validate_text_content(
        cls, value: str | None, info: ValidationInfo
    ) -> str | None:
        node_type: NodeType = info.data.get("node_type")
        if node_type == NodeType.TEXT:
            if value is None:
                raise ValueError(
                    "NodeType == 'text', but text_content is None."
                )

        if node_type == NodeType.MULTIMODAL:
            if value is None:
                raise ValueError(
                    "NodeType == 'multimodal', but text_content is None."
                )

        return value

    @field_validator("image_content", mode="after")
    @classmethod
    def validate_image_content(
        cls, value: str | None, info: ValidationInfo
    ) -> str | None:
        node_type: NodeType = info.data.get("node_type")
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
