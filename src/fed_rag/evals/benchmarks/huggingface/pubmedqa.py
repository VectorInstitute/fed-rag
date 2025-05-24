"""PubMedQA benchmark"""

from typing import Any, ClassVar

from pydantic import model_validator

from fed_rag.base.evals.benchmark import BaseBenchmark

from .mixin import HuggingFaceBenchmarkMixin
from .utils import check_huggingface_evals_installed


class HuggingFacePubMedQA(HuggingFaceBenchmarkMixin, BaseBenchmark):
    """HuggingFace PubMedQA Benchmark.

    PubMedQA is a biomedical question answering dataset where each question
    can be answered with "yes", "no", or "maybe" based on the given context.

    Example schema:
        {
            "pubid": "25429730",
            "question": "Are group 2 innate lymphoid cells ( ILC2s ) increased in chronic rhinosinusitis with nasal polyps or eosinophilia?",
            "context": "...",  # Abstract text
            "long_answer": "...",  # Detailed answer
            "final_decision": "yes"  # or "no" or "maybe"
        }
    """

    dataset_name = "qiaojin/PubMedQA"
    configuration_name: str = "pqa_labeled"
    response_key: ClassVar[dict[str, str]] = {
        "yes": "yes",
        "no": "no",
        "maybe": "maybe",
    }

    def _get_query_from_example(self, example: dict[str, Any]) -> str:
        return str(example["question"])

    def _get_response_from_example(self, example: dict[str, Any]) -> str:
        final_decision = example["final_decision"]
        return self.response_key.get(final_decision, final_decision)

    def _get_context_from_example(self, example: dict[str, Any]) -> str | None:
        context = example.get("context", {})
        if isinstance(context, dict):
            return " ".join(context.values())
        elif isinstance(context, list):
            return " ".join(context)
        elif isinstance(context, str):
            return context
        return None

    @model_validator(mode="before")
    @classmethod
    def _validate_extra_installed(cls, data: Any) -> Any:
        """Validate that huggingface-evals dependencies are installed."""
        check_huggingface_evals_installed(cls.__name__)
        return data
