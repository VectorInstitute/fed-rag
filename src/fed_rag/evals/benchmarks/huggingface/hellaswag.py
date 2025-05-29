"""HellaSwag benchmark."""

from typing import Any

from pydantic import model_validator

from fed_rag.base.evals.benchmark import BaseBenchmark

from .mixin import HuggingFaceBenchmarkMixin
from .utils import check_huggingface_evals_installed


class HuggingFaceHellaSwag(HuggingFaceBenchmarkMixin, BaseBenchmark):
    """HuggingFace HellaSwag Benchmark.

    HellaSwag is a commonsense reasoning dataset where each example consists of a context
    and four possible endings. The task is to pick the most plausible ending.

    Example schema:
        {
            "ind": 4,
            "activity_label": "Removing ice from car",
            "ctx_a": "Then, the man writes over the snow covering the window of a car, and a woman wearing winter clothes smiles.",
            "ctx_b": "then",
            "ctx": "Then, the man writes over the snow covering the window of a car, and a woman wearing winter clothes smiles. then",
            "endings": [
                ", the man adds wax to the windshield and cuts it.",
                ", a person board a ski lift, while two men supporting the head of the person wearing winter clothes snow as the we girls sled.",
                ", the man puts on a christmas coat, knitted with netting.",
                ", the man continues removing the snow on his car."
            ],
            "source_id": "activitynet~v_-1IBHYS3L-Y",
            "split": "train",
            "split_type": "indomain",
            "label": "3"
        }
    """

    dataset_name = "Rowan/hellaswag"

    def _get_query_from_example(self, example: dict[str, Any]) -> str:
        # Use the full context for the prompt
        return str(example["ctx"])

    def _get_response_from_example(self, example: dict[str, Any]) -> str:
        # The correct ending index as string or int
        return str(example["label"])

    def _get_context_from_example(self, example: dict[str, Any]) -> str:
        # Show the four endings as context (choices)
        if "endings" in example and isinstance(example["endings"], list):
            return "\n".join(
                f"{i}: {ending.strip()}"
                for i, ending in enumerate(example["endings"])
            )
        return ""

    @model_validator(mode="before")
    @classmethod
    def _validate_extra_installed(cls, data: Any) -> Any:
        """Validate that huggingface-evals dependencies are installed."""
        check_huggingface_evals_installed(cls.__name__)
        return data
