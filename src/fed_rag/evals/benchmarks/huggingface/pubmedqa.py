"""PubMedQA benchmark"""

from typing import Any

from pydantic import model_validator

from fed_rag.base.evals.benchmark import BaseBenchmark
from fed_rag.exceptions import BenchmarkParseError

from .mixin import HuggingFaceBenchmarkMixin
from .utils import check_huggingface_evals_installed


class HuggingFacePubMedQA(HuggingFaceBenchmarkMixin, BaseBenchmark):
    """HuggingFace PubMedQA Benchmark.

    PubMedQA is a biomedical question answering dataset where each question
    can be answered with "yes", "no", or "maybe" based on the given context.

    Example schema:
        {
            "pubid": "25429730",
            "question": "Are group 2 innate lymphoid cells (ILC2s) increased in chronic rhinosinusitis with nasal polyps or eosinophilia?",
            "context": {
                "contexts": [
                    "Chronic rhinosinusitis is a heterogeneous disease with uncertain pathogenesis.",
                    "The study aimed to identify ILC2s in sinus mucosa in patients with CRS.",
                    "35 patients including 13 with eosinophilic CRS were recruited.",
                    "ILC2 frequencies were associated with the presence of nasal polyps and increased blood eosinophilia."
                ],
                "labels": ["label1", "label2", "label3", "label4"],
                "meshes": ["Chronic Disease", "Nasal Polyps", "Immunity, Innate"]
            },
            "long_answer": "Based on our analysis, increased ILC2s are associated with CRS with nasal polyps.",
            "final_decision": "yes"  # or "no" or "maybe"
        }"""

    dataset_name = "qiaojin/PubMedQA"
    configuration_name: str = "pqa_labeled"

    def _get_query_from_example(self, example: dict[str, Any]) -> str:
        return str(example["question"])

    def _get_response_from_example(self, example: dict[str, Any]) -> str:
        return str(example["final_decision"])

    def _get_context_from_example(self, example: dict[str, Any]) -> str:
        context = example.get("context", {})
        if isinstance(context, dict):
            contexts_list = context.get("contexts")
            if isinstance(contexts_list, list):
                return " ".join(contexts_list)
            # Fallback: join all values if "contexts" is missing
            return " ".join(
                " ".join(v) if isinstance(v, list) else str(v)
                for v in context.values()
            )
        elif isinstance(context, str):
            return context
        else:
            raise BenchmarkParseError(
                f"Unexpected context type: {type(context)} in example: {example}"
            )

    @model_validator(mode="before")
    @classmethod
    def _validate_extra_installed(cls, data: Any) -> Any:
        """Validate that huggingface-evals dependencies are installed."""
        check_huggingface_evals_installed(cls.__name__)
        return data
