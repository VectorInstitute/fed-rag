"""SQuAD 2.0 benchmark"""

from typing import Any

from pydantic import model_validator

from fed_rag.base.evals.benchmark import BaseBenchmark

from .mixin import HuggingFaceBenchmarkMixin
from .utils import check_huggingface_evals_installed


class HuggingFaceSQuADv2(HuggingFaceBenchmarkMixin, BaseBenchmark):
    """HuggingFace SQuAD 2.0 Benchmark.

    Stanford Question Answering Dataset (SQuAD) 2.0 combines 100,000 questions
    from SQuAD 1.1 with over 50,000 unanswerable questions. Systems must not
    only answer questions when possible, but also determine when no answer is
    supported by the paragraph.

    Example schema:
        {
            "id": "56ddde2d66d3e219004dad4d",
            "title": "Symbiosis",
            "context": "Symbiotic relationships include those associations in which one organism lives on another (ectosymbiosis, such as...",
            "question": "What is an example of ectosymbiosis?",
            "answers": {
                "text": ["mistletoe"],
                "answer_start": [114]
            }
        }

    For unanswerable questions, the answers field has empty lists:
        {
            "answers": {
                "text": [],
                "answer_start": []
            }
        }
    """

    dataset_name = "squad_v2"
    configuration_name: str | None = None

    def _get_query_from_example(self, example: dict[str, Any]) -> str:
        return str(example["question"])

    def _get_response_from_example(self, example: dict[str, Any]) -> str:
        answers = example.get("answers", {})
        answer_texts = answers.get("text", [])

        if answer_texts:
            # Return the first answer (they are typically variations of the same answer)
            return str(answer_texts[0])
        else:
            # For unanswerable questions, return a special token
            return "[NO ANSWER]"

    def _get_context_from_example(self, example: dict[str, Any]) -> str:
        return str(example["context"])

    @model_validator(mode="before")
    @classmethod
    def _validate_extra_installed(cls, data: Any) -> Any:
        """Validate that huggingface-evals dependencies are installed."""
        check_huggingface_evals_installed(cls.__name__)
        return data
