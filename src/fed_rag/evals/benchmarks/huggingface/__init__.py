from .boolq import HuggingFaceBoolQ
from .hellaswag import HuggingFaceHellaSwag
from .hotpotqa import HuggingFaceHotpotQA
from .mixin import HuggingFaceBenchmarkMixin
from .mmlu import HuggingFaceMMLU
from .natural_questions import HuggingFaceNaturalQuestions
from .pubmedqa import HuggingFacePubMedQA
from .squad_v2 import HuggingFaceSQuADv2

__all__ = [
    "HuggingFaceBenchmarkMixin",
    "HuggingFaceMMLU",
    "HuggingFacePubMedQA",
    "HuggingFaceHotpotQA",
    "HuggingFaceSQuADv2",
    "HuggingFaceNaturalQuestions",
    "HuggingFaceBoolQ",
    "HuggingFaceHellaSwag",
]
