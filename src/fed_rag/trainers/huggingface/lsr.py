"""HuggingFace LM-Supervised Retriever Trainer"""

from typing import TYPE_CHECKING, Any

from fed_rag.base.trainer import BaseTrainer
from fed_rag.exceptions import TrainerError
from fed_rag.trainers.huggingface.mixin import HuggingFaceTrainerMixin

if TYPE_CHECKING:  # pragma: no-cover
    from sentence_transformers import SentenceTransformer


class HuggingFaceLSRTrainer(HuggingFaceTrainerMixin, BaseTrainer):
    """HuggingFace LM-Supervised Retriever Trainer."""

    model: "SentenceTransformer"

    def __init__(self, model: SentenceTransformer, **kwargs: Any):
        # check that the model is a `SentenceTransformer`
        if not isinstance(model, SentenceTransformer):
            raise TrainerError(
                "For `HuggingFaceLSRTrainer`, attribute `model` must be of type "
                "`~sentence_transformers.SentenceTransformer`."
            )

        super().__init__(model=model, **kwargs)
