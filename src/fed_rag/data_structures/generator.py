"""Auxiliary types for RAG System"""

from typing import Any

from PIL import Image
from pydantic import BaseModel, ConfigDict


class _MultiModalDataContainer(BaseModel):
    """A multi-modal data container.

    This class represents a multimodal representation of a RAG query.

    Attributes:
        text: Text content of query.
        images: Images content of query.
        audios: Audios content of query.
        videos: Videos content of query.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    text: str
    images: list[Image.Image] | None = None
    audios: list[Any] | None = None
    videos: list[Any] | None = None

    def __str__(self) -> str:
        return self.text


class Query(_MultiModalDataContainer):
    """Query data structure.

    This class represents a multimodal representation of a RAG query.

    Attributes:
        text: Text content of query.
        images: Images content of query.
        audios: Audios content of query.
        videos: Videos content of query.
    """

    pass


class Context(_MultiModalDataContainer):
    """Context data structure.

    This class represents a multimodal representation of RAG context.

    Attributes:
        text: Text content of query.
        images: Images content of query.
        audios: Audios content of query.
        videos: Videos content of query.
    """

    pass


class Prompt(_MultiModalDataContainer):
    """Prompt data structure.

    This class represents a multimodal representation of a prompt given to a
    multi-modal LLM.

    Attributes:
        text: Text content of query.
        images: Images content of query.
        audios: Audios content of query.
        videos: Videos content of query.
    """

    pass
