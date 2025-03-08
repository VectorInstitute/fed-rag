"""Knowledge Node"""

import uuid
from enum import Enum

from pydantic import BaseModel, Field


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
    metadata: dict = Field(
        description="Metadata for node.", default_factory=dict
    )
    text_content: str | None = Field(
        description="Text content. Used for TEXT and potentially MULTIMODAL node types."
    )
    image_content: bytes | None = Field(
        description="Image content as binary data (decoded from base64)"
    )

    def get_content(self) -> str | bytes | dict | None:
        """Return the appropriate content based on node_type."""
        if self.node_type == NodeType.TEXT:
            return self.text_content
        elif self.node_type == NodeType.IMAGE:
            return self.image_content
        elif self.node_type == NodeType.MULTIMODAL:
            return {"text": self.text_content, "image": self.image_content}
