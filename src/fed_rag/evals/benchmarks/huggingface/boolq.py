"""BoolQ benchmark"""

from typing import Any

from pydantic import model_validator

from fed_rag.base.evals.benchmark import BaseBenchmark

from .mixin import HuggingFaceBenchmarkMixin
from .utils import check_huggingface_evals_installed


class HuggingFaceBoolQ(HuggingFaceBenchmarkMixin, BaseBenchmark):
    """HuggingFace BoolQ Benchmark.

    BoolQ is a question answering dataset for yes/no questions about a short passage.

    Example schema:
        {
            "question": "does ethanol take more energy make that produces",
            "answer": false,
            "passage": "\"All biomass goes through at least some of these steps: ...",
        }
    """

    dataset_name = "google/boolq"

    def _get_query_from_example(self, example: dict[str, Any]) -> str:
        return str(example["question"])

    def _get_response_from_example(self, example: dict[str, Any]) -> str:
        # Return as string "true"/"false" for consistency
        return "true" if example["answer"] else "false"

    def _get_context_from_example(self, example: dict[str, Any]) -> str:
        return str(example["passage"])

    @model_validator(mode="before")
    @classmethod
    def _validate_extra_installed(cls, data: Any) -> Any:
        check_huggingface_evals_installed(cls.__name__)
        return data
