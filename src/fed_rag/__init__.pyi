"""Type stubs for fed_rag module"""

import types

from .generators import (
    HFPeftModelGenerator,
    HFPretrainedModelGenerator,
    UnslothFastModelGenerator,
)
from .retrievers import HFSentenceTransformerRetriever
from .trainer_managers import (
    HuggingFaceRAGTrainerManager,
    PyTorchRAGTrainerManager,
)
from .trainers import HuggingFaceTrainerForLSR, HuggingFaceTrainerForRALT

generators: types.ModuleType
retrievers: types.ModuleType
trainers: types.ModuleType
trainer_managers: types.ModuleType

__all__ = [
    "HFPeftModelGenerator",
    "HFPretrainedModelGenerator",
    "UnslothFastModelGenerator",
    "HFSentenceTransformerRetriever",
    "HuggingFaceRAGTrainerManager",
    "PyTorchRAGTrainerManager",
    "HuggingFaceTrainerForLSR",
    "HuggingFaceTrainerForRALT",
]
