"""MMLU benchmark"""

from typing import Any

from pydantic import model_validator

from fed_rag.base.evals.benchmark import BaseBenchmark

from .mixin import HuggingFaceBenchmarkMixin
from .utils import check_huggingface_evals_installed


class HuggingFaceMMLU(HuggingFaceBenchmarkMixin, BaseBenchmark):
    """HuggingFace MMLU Benchmark.

    Example schema:
        {
            "question": "What is the embryological origin of the hyoid bone?",
            "choices": [
                "The first pharyngeal arch",
                "The first and second pharyngeal arches",
                "The second pharyngeal arch",
                "The second and third pharyngeal arches",
            ],
            "answer": "D",
        }
    """

    dataset_name = "cais/mmlu"
    configuration_name: str = "all"

    def _get_query_from_example(self, example: dict[str, Any]) -> str:
        choices = example["choices"]
        formatted_choices = f"A: {choices[0]}\nB: {choices[1]}\nC: {choices[2]}\nD: {choices[3]}"
        return f"{example['question']}\n\n{formatted_choices}"

    def _get_response_from_example(self, example: dict[str, Any]) -> str:
        return str(example["answer"])

    def _get_context_from_example(self, example: dict[str, Any]) -> str | None:
        return None

    @model_validator(mode="before")
    @classmethod
    def _validate_extra_installed(cls, data: Any) -> Any:
        """Validate that huggingface-evals dependencies are installed."""
        check_huggingface_evals_installed(cls.__name__)
        return data
