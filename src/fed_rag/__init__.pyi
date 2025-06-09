"""Type stubs for fed_rag module"""

# Lazy-loaded classes (type-only declarations)
# Lazy-loaded modules
from fed_rag import generators as generators
from fed_rag import retrievers as retrievers
from fed_rag import trainer_managers as trainer_managers
from fed_rag import trainers as trainers
from fed_rag.generators import HFPeftModelGenerator as HFPeftModelGenerator
from fed_rag.generators import (
    HFPretrainedModelGenerator as HFPretrainedModelGenerator,
)
from fed_rag.generators import (
    UnslothFastModelGenerator as UnslothFastModelGenerator,
)
from fed_rag.retrievers import (
    HFSentenceTransformerRetriever as HFSentenceTransformerRetriever,
)
from fed_rag.trainer_managers import (
    HuggingFaceRAGTrainerManager as HuggingFaceRAGTrainerManager,
)
from fed_rag.trainer_managers import (
    PyTorchRAGTrainerManager as PyTorchRAGTrainerManager,
)
from fed_rag.trainers import (
    HuggingFaceTrainerForLSR as HuggingFaceTrainerForLSR,
)
from fed_rag.trainers import (
    HuggingFaceTrainerForRALT as HuggingFaceTrainerForRALT,
)
