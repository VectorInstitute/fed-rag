"""MMLU benchmark"""

from typing import Any

from pydantic import model_validator

from fed_rag.base.evals.benchmark import BaseBenchmark

from .mixin import HuggingFaceBenchmarkMixin
from .utils import check_huggingface_evals_installed

# {
#     "question": "What is the embryological origin of the hyoid bone?",
#     "choices": [
#         "The first pharyngeal arch",
#         "The first and second pharyngeal arches",
#         "The second pharyngeal arch",
#         "The second and third pharyngeal arches",
#     ],
#     "answer": "D",
# }


class HuggingFaceMMLU(HuggingFaceBenchmarkMixin, BaseBenchmark):
    """HuggingFace MMLU Benchmark."""

    dataset_name = "cais/mmlu"
    configuration_name: str = "all"

    def _get_query_from_example(self, example: dict[str, Any]) -> str:
        raise NotImplementedError

    def _get_response_from_example(self, example: dict[str, Any]) -> str:
        raise NotImplementedError

    def _get_context_from_example(self, example: dict[str, Any]) -> str:
        raise NotImplementedError

    @model_validator(mode="before")
    @classmethod
    def _validate_extra_installed(cls, data: Any) -> Any:
        """Validate that huggingface-evals dependencies are installed."""
        check_huggingface_evals_installed(cls.__name__)
        return data
