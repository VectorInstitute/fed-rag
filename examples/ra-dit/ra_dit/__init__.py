from .logger import logger

CHECKPOINT_DIR_TEMPLATES = {
    "generator": (
        ".checkpoints/{retriever_id}-{generator_id}-{generator_variant}/generator"
    ),
    "retriever": (
        ".checkpoints/{retriever_id}-{generator_id}-{generator_variant}/retriever"
    ),
}

__all__ = ["logger"]
