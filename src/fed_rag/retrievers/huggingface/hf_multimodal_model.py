"""HF Multimodal Retriever """

from typing import TYPE_CHECKING, Any, Literal, Optional, Sequence, cast

import numpy as np
import torch
from PIL import Image as PILImage
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

try:
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModel, AutoProcessor, PreTrainedModel

    _has_huggingface = True
except ModuleNotFoundError:
    _has_huggingface = False

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer
    from transformers import PreTrainedModel, AutoModel, AutoProcessor

from fed_rag.base.generator_mixins.audio import AudioModalityMixin
from fed_rag.base.generator_mixins.image import ImageModalityMixin
from fed_rag.base.generator_mixins.video import VideoModalityMixin
from fed_rag.base.retriever import BaseRetriever
from fed_rag.data_structures.rag import Context, Query
from fed_rag.exceptions import MissingExtraError, RetrieverError


class LoadKwargs(BaseModel):
    encoder: dict = Field(default_factory=dict)
    query_encoder: dict = Field(default_factory=dict)
    context_encoder: dict = Field(default_factory=dict)
    text_encoder: dict = Field(default_factory=dict)
    image_encoder: dict = Field(default_factory=dict)
    audio_encoder: dict = Field(default_factory=dict)
    video_encoder: dict = Field(default_factory=dict)


class InvalidLoadType(RetrieverError):
    """Raised if an invalid load type was supplied."""

    pass


class HFMultimodalModelRetriever(
    ImageModalityMixin,
    AudioModalityMixin,
    VideoModalityMixin,
    BaseRetriever,
):
    model_config = ConfigDict(
        protected_namespaces=("pydantic_model_",), arbitrary_types_allowed=True
    )
    model_name: str | None = Field(
        description="Name of HuggingFace multimodal model.",
        default=None,
    )
    query_model_name: str | None = Field(
        description="Name of HuggingFace model used for encoding queries.",
        default=None,
    )
    context_model_name: str | None = Field(
        description="Name of HuggingFace model used for encoding context.",
        default=None,
    )
    text_model_name: str | None = Field(
        description="Name of HuggingFace SentenceTransformer model used for encoding text.",
        default=None,
    )
    image_model_name: str | None = Field(
        description="Name of HuggingFace model used for encoding images.",
        default=None,
    )
    audio_model_name: str | None = Field(
        description="Name of HuggingFace model used for encoding audio.",
        default=None,
    )
    video_model_name: str | None = Field(
        description="Name of HuggingFace model used for encoding video.",
        default=None,
    )
    load_model_kwargs: LoadKwargs = Field(
        description="Optional kwargs dict for loading models from HF. Defaults to None.",
        default_factory=LoadKwargs,
    )

    _encoder: Optional["PreTrainedModel | SentenceTransformer"] = PrivateAttr(
        default=None
    )
    _query_encoder: Optional[
        "PreTrainedModel | SentenceTransformer"
    ] = PrivateAttr(default=None)
    _context_encoder: Optional[
        "PreTrainedModel | SentenceTransformer"
    ] = PrivateAttr(default=None)
    _text_encoder: Optional["SentenceTransformer"] = PrivateAttr(default=None)
    _image_encoder: Optional["PreTrainedModel"] = PrivateAttr(default=None)
    _audio_encoder: Optional["PreTrainedModel"] = PrivateAttr(default=None)
    _video_encoder: Optional["PreTrainedModel"] = PrivateAttr(default=None)

    def __init__(
        self,
        model_name: str | None = None,
        query_model_name: str | None = None,
        context_model_name: str | None = None,
        text_model_name: str | None = None,
        image_model_name: str | None = None,
        audio_model_name: str | None = None,
        video_model_name: str | None = None,
        load_model_kwargs: LoadKwargs | dict | None = None,
        load_model_at_init: bool = True,
    ):
        if not _has_huggingface:
            msg = (
                f"`{self.__class__.__name__}` requires `huggingface` extra to be installed. "
                "To fix please run `pip install fed-rag[huggingface]`."
            )
            raise MissingExtraError(msg)

        if isinstance(load_model_kwargs, dict):
            # use same dict for all
            load_model_kwargs = LoadKwargs(
                encoder=load_model_kwargs,
                query_encoder=load_model_kwargs,
                context_encoder=load_model_kwargs,
                text_encoder=load_model_kwargs,
                image_encoder=load_model_kwargs,
                audio_encoder=load_model_kwargs,
                video_encoder=load_model_kwargs,
            )

        load_model_kwargs = (
            load_model_kwargs if load_model_kwargs else LoadKwargs()
        )

        super().__init__(
            model_name=model_name,
            query_model_name=query_model_name,
            context_model_name=context_model_name,
            text_model_name=text_model_name,
            image_model_name=image_model_name,
            audio_model_name=audio_model_name,
            video_model_name=video_model_name,
            load_model_kwargs=load_model_kwargs,
        )
        if load_model_at_init:
            if model_name:
                self._encoder = self._load_model_from_hf(load_type="encoder")
            else:
                if self.query_model_name:
                    self._query_encoder = cast(
                        "PreTrainedModel | SentenceTransformer",
                        self._load_model_from_hf(load_type="query_encoder"),
                    )
                if self.context_model_name:
                    self._context_encoder = cast(
                        "PreTrainedModel | SentenceTransformer",
                        self._load_model_from_hf(load_type="context_encoder"),
                    )
                if self.text_model_name:
                    self._text_encoder = cast(
                        "SentenceTransformer",
                        self._load_model_from_hf(load_type="text_encoder"),
                    )
                if self.image_model_name:
                    self._image_encoder = cast(
                        "PreTrainedModel",
                        self._load_model_from_hf(load_type="image_encoder"),
                    )
                if self.audio_model_name:
                    self._audio_encoder = cast(
                        "PreTrainedModel",
                        self._load_model_from_hf(load_type="audio_encoder"),
                    )
                if self.video_model_name:
                    self._video_encoder = cast(
                        "PreTrainedModel",
                        self._load_model_from_hf(load_type="video_encoder"),
                    )

    def _load_model_from_hf(
        self,
        load_type: Literal[
            "encoder",
            "query_encoder",
            "context_encoder",
            "text_encoder",
            "image_encoder",
            "audio_encoder",
            "video_encoder",
        ],
        **kwargs: Any,
    ) -> "PreTrainedModel | SentenceTransformer":
        # Mapping of load_type to (model_name, load_kwargs_key, model_class)
        load_configs = {
            "encoder": (self.model_name, "encoder", AutoModel),
            "query_encoder": (
                self.query_model_name,
                "query_encoder",
                AutoModel,
            ),
            "context_encoder": (
                self.context_model_name,
                "context_encoder",
                AutoModel,
            ),
            "text_encoder": (
                self.text_model_name,
                "text_encoder",
                SentenceTransformer,
            ),
            "image_encoder": (
                self.image_model_name,
                "image_encoder",
                AutoModel,
            ),
            "audio_encoder": (
                self.audio_model_name,
                "audio_encoder",
                AutoModel,
            ),
            "video_encoder": (
                self.video_model_name,
                "video_encoder",
                AutoModel,
            ),
        }

        if load_type not in load_configs:
            raise InvalidLoadType("Invalid `load_type` supplied.")

        model_name, kwargs_key, model_class = load_configs[load_type]
        load_kwargs = getattr(self.load_model_kwargs, kwargs_key).copy()
        load_kwargs.update(kwargs)

        if model_name is None:
            raise RetrieverError(f"No model name specified for {load_type}")

        if model_class == SentenceTransformer:
            return cast("SentenceTransformer", model_class.from_pretrained(model_name, **load_kwargs))  # type: ignore[attr-defined]
        else:
            return cast("PreTrainedModel", model_class.from_pretrained(model_name, **load_kwargs))  # type: ignore[attr-defined]

    def _detect_content_type(self, item: Any) -> tuple[str, Any]:
        """Detect content type and return (content_type, processed_item)."""
        # Try to detect content type
        try:
            if isinstance(item, PILImage.Image):
                return "image", item
        except ImportError:
            pass

        # Check if it's a numpy array (could be image, audio, or video)
        if hasattr(item, "shape") and hasattr(item, "dtype"):
            try:
                if len(item.shape) == 3 and item.shape[2] in [
                    1,
                    3,
                    4,
                ]:  # Likely an image
                    if item.dtype == np.uint8:
                        pil_image = PILImage.fromarray(item)
                        return "image", pil_image
            except (ImportError, ValueError, TypeError):
                pass

        # For other types, convert to string as fallback
        return "text", str(item)

    def _create_query(self, q: str | Query) -> Query:
        """Convert input to Query object with proper content type detection."""
        if isinstance(q, Query):
            return q
        elif isinstance(q, str):
            return Query(text=q)
        else:
            content_type, processed_item = self._detect_content_type(q)
            if content_type == "image":
                return Query(text="", images=[processed_item])
            else:
                return Query(text=processed_item)

    def _create_context(self, c: str | Context) -> Context:
        """Convert input to Context object with proper content type detection."""
        if isinstance(c, Context):
            return c
        elif isinstance(c, str):
            return Context(text=c)
        else:
            content_type, processed_item = self._detect_content_type(c)
            if content_type == "image":
                return Context(text="", images=[processed_item])
            else:
                return Context(text=processed_item)

    def _process_content(
        self, items: Sequence[Query | Context]
    ) -> list[dict[str, Any]]:
        """Process items into a unified content list."""
        content: list[dict[str, Any]] = []
        for item in items:
            if item is not None:
                if getattr(item, "text", None):
                    content.append({"type": "text", "text": item.text})
                for im in getattr(item, "images", []) or []:
                    if isinstance(im, np.ndarray):
                        im = PILImage.fromarray(im)
                    content.append({"type": "image", "image": im})
                for au in getattr(item, "audios", []) or []:
                    content.append({"type": "audio", "audio": au})
                for vi in getattr(item, "videos", []) or []:
                    content.append({"type": "video", "video": vi})
        return content

    def _encode_modalities(
        self, content: list[dict[str, Any]], **kwargs: Any
    ) -> list[torch.Tensor]:
        """Encode all modalities present in content."""
        embeddings = []

        # Check for different modalities
        has_images = any(item["type"] == "image" for item in content)
        has_audios = any(item["type"] == "audio" for item in content)
        has_videos = any(item["type"] == "video" for item in content)
        has_text = any(item["type"] == "text" for item in content)

        # Encode images if present
        if (
            has_images
            and self.image_model_name
            and self.image_model is not None
        ):
            all_images = [
                item["image"] for item in content if item["type"] == "image"
            ]
            if all_images:
                img_emb = self.encode_image(all_images, **kwargs)
                embeddings.append(img_emb)

        # Encode audio if present
        if (
            has_audios
            and self.audio_model_name
            and self.audio_model is not None
        ):
            all_audios = [
                item["audio"] for item in content if item["type"] == "audio"
            ]
            if all_audios:
                audio_emb = self.encode_audio(all_audios, **kwargs)
                embeddings.append(audio_emb)

        # Encode video if present
        if (
            has_videos
            and self.video_model_name
            and self.video_model is not None
        ):
            all_videos = [
                item["video"] for item in content if item["type"] == "video"
            ]
            if all_videos:
                video_emb = self.encode_video(all_videos, **kwargs)
                embeddings.append(video_emb)

        # Encode text if present
        if has_text and self.text_model_name and self.text_model is not None:
            all_text = [
                item["text"] for item in content if item["type"] == "text"
            ]
            if all_text:
                text_emb = self.encode_text(all_text, **kwargs)
                embeddings.append(text_emb)

        return embeddings

    def encode_query(
        self, query: str | list[str] | Query | list[Query], **kwargs: Any
    ) -> torch.Tensor:
        # Handle list of queries
        if isinstance(query, list):
            queries: Sequence[Query] = [self._create_query(q) for q in query]
        else:
            queries = [self._create_query(query)]

        # Process content and encode modalities
        content = self._process_content(queries)
        embeddings = self._encode_modalities(content, **kwargs)

        # Combine embeddings if we have multiple modalities
        if len(embeddings) > 1:
            return torch.cat(embeddings, dim=-1)
        elif len(embeddings) == 1:
            return embeddings[0]

        # Fall back to general encoding if no modality-specific models
        if self.query_encoder is not None:
            if hasattr(self.query_encoder, "encode"):
                # Convert Query objects to text for encoding
                text_queries = [q.text for q in queries if q.text]
                if text_queries:
                    result = self.query_encoder.encode(text_queries)
                    if isinstance(result, np.ndarray):
                        return torch.from_numpy(result)
                    return cast(torch.Tensor, result)
                else:
                    raise RetrieverError(
                        "No text content in queries for encoding"
                    )
            else:
                raise RetrieverError(
                    "Query encoder does not have encode method"
                )
        elif self.encoder is not None:
            if hasattr(self.encoder, "encode"):
                # Convert Query objects to text for encoding
                text_queries = [q.text for q in queries if q.text]
                if text_queries:
                    result = self.encoder.encode(text_queries)
                    if isinstance(result, np.ndarray):
                        return torch.from_numpy(result)
                    return cast(torch.Tensor, result)
                else:
                    raise RetrieverError(
                        "No text content in queries for encoding"
                    )
            else:
                raise RetrieverError("Encoder does not have encode method")
        else:
            raise RetrieverError("No query encoder available for encoding")

    def encode_context(
        self, context: str | list[str] | Context | list[Context], **kwargs: Any
    ) -> torch.Tensor:
        # Handle list of contexts
        if isinstance(context, list):
            contexts: Sequence[Context] = [
                self._create_context(c) for c in context
            ]
        else:
            contexts = [self._create_context(context)]

        # Process content and encode modalities
        content = self._process_content(contexts)
        embeddings = self._encode_modalities(content, **kwargs)

        # Combine embeddings if we have multiple modalities
        if len(embeddings) > 1:
            return torch.cat(embeddings, dim=-1)
        elif len(embeddings) == 1:
            return embeddings[0]

        # Fall back to general encoding if no modality-specific models
        if self.context_encoder is not None:
            if hasattr(self.context_encoder, "encode"):
                # Convert Context objects to text for encoding
                text_contexts = [c.text for c in contexts if c.text]
                if text_contexts:
                    result = self.context_encoder.encode(text_contexts)
                    if isinstance(result, np.ndarray):
                        return torch.from_numpy(result)
                    return cast(torch.Tensor, result)
                else:
                    raise RetrieverError(
                        "No text content in contexts for encoding"
                    )
            else:
                raise RetrieverError(
                    "Context encoder does not have encode method"
                )
        elif self.encoder is not None:
            if hasattr(self.encoder, "encode"):
                # Convert Context objects to text for encoding
                text_contexts = [c.text for c in contexts if c.text]
                if text_contexts:
                    result = self.encoder.encode(text_contexts)
                    if isinstance(result, np.ndarray):
                        return torch.from_numpy(result)
                    return cast(torch.Tensor, result)
                else:
                    raise RetrieverError(
                        "No text content in contexts for encoding"
                    )
            else:
                raise RetrieverError("Encoder does not have encode method")
        else:
            raise RetrieverError("No context encoder available for encoding")

    def encode_text(
        self, text: list[str] | list[Any], **kwargs: Any
    ) -> torch.Tensor:
        encoder = self.text_model if self.text_model else self._text_encoder
        if encoder is None:
            raise RetrieverError("No text encoder available")
        result = cast(SentenceTransformer, encoder).encode(text)
        if isinstance(result, np.ndarray):
            return torch.from_numpy(result)
        return cast(torch.Tensor, result)

    def _get_processor(self, model_name: str) -> Any:
        return AutoProcessor.from_pretrained(model_name)

    def _extract_embeddings(self, outputs: Any) -> torch.Tensor:
        if hasattr(outputs, "last_hidden_state"):
            emb = outputs.last_hidden_state
        elif hasattr(outputs, "pooler_output"):
            emb = outputs.pooler_output
        else:
            raise RetrieverError("No embeddings found")

        if isinstance(emb, torch.Tensor):
            return emb.mean(dim=1) if emb.ndim > 2 else emb
        else:
            raise RetrieverError("Embeddings are not a torch.Tensor")

    def _encode(
        self, data: Any, model: Any, model_name: str, **kwargs: Any
    ) -> torch.Tensor:
        if model is None:
            raise RetrieverError("No encoder available")
        processor = self._get_processor(model_name)
        inputs = processor(data, return_tensors="pt", **kwargs)
        with torch.inference_mode():
            outputs = model(**inputs, output_hidden_states=True)
        return self._extract_embeddings(outputs)

    def encode_image(self, images: Any, **kwargs: Any) -> torch.Tensor:
        if self.image_model_name is None:
            raise RetrieverError("No image model name specified")
        return self._encode(
            images, self.image_model, self.image_model_name, **kwargs
        )

    def encode_audio(self, audios: Any, **kwargs: Any) -> torch.Tensor:
        if self.audio_model_name is None:
            raise RetrieverError("No audio model name specified")
        return self._encode(
            audios, self.audio_model, self.audio_model_name, **kwargs
        )

    def encode_video(self, videos: Any, **kwargs: Any) -> torch.Tensor:
        if self.video_model_name is None:
            raise RetrieverError("No video model name specified")
        return self._encode(
            videos, self.video_model, self.video_model_name, **kwargs
        )

    def _get_or_load_model(
        self,
        model_name: str | None,
        private_attr: str,
        load_type: Literal[
            "encoder",
            "query_encoder",
            "context_encoder",
            "text_encoder",
            "image_encoder",
            "audio_encoder",
            "video_encoder",
        ],
    ) -> Any:
        """Get model from private attribute or load it if not available."""
        if model_name and getattr(self, private_attr) is None:
            setattr(
                self,
                private_attr,
                self._load_model_from_hf(load_type=load_type),
            )
        return getattr(self, private_attr)

    @property
    def image_model(self) -> Optional["PreTrainedModel"]:
        result = self._get_or_load_model(
            self.image_model_name, "_image_encoder", "image_encoder"
        )
        return cast(Optional["PreTrainedModel"], result)

    @property
    def audio_model(self) -> Optional["PreTrainedModel"]:
        result = self._get_or_load_model(
            self.audio_model_name, "_audio_encoder", "audio_encoder"
        )
        return cast(Optional["PreTrainedModel"], result)

    @property
    def video_model(self) -> Optional["PreTrainedModel"]:
        result = self._get_or_load_model(
            self.video_model_name, "_video_encoder", "video_encoder"
        )
        return cast(Optional["PreTrainedModel"], result)

    @property
    def text_model(self) -> Optional["SentenceTransformer"]:
        result = self._get_or_load_model(
            self.text_model_name, "_text_encoder", "text_encoder"
        )
        return cast(Optional["SentenceTransformer"], result)

    @property
    def encoder(self) -> Optional["PreTrainedModel | SentenceTransformer"]:
        result = self._get_or_load_model(
            self.model_name, "_encoder", "encoder"
        )
        return cast(Optional["PreTrainedModel | SentenceTransformer"], result)

    @property
    def query_encoder(
        self,
    ) -> Optional["PreTrainedModel | SentenceTransformer"]:
        result = self._get_or_load_model(
            self.query_model_name, "_query_encoder", "query_encoder"
        )
        return cast(Optional["PreTrainedModel | SentenceTransformer"], result)

    @property
    def context_encoder(
        self,
    ) -> Optional["PreTrainedModel | SentenceTransformer"]:
        result = self._get_or_load_model(
            self.context_model_name, "_context_encoder", "context_encoder"
        )
        return cast(Optional["PreTrainedModel | SentenceTransformer"], result)
