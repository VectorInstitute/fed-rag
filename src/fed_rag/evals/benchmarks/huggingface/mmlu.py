"""MMLU benchmark"""

from typing import Any, ClassVar

from pydantic import Field, model_validator

from fed_rag.base.evals.benchmark import BaseBenchmark

from .mixin import HuggingFaceBenchmarkMixin
from .utils import check_huggingface_evals_installed

DEFAULT_MMLU_PROMPT = """
    {question}
    A. {A}
    B. {B}
    C. {C}
    D. {D}

    You MUST answer ONLY with a single letter (A, B, C, or D). Do NOT provide any explanation, reasoning, or extra text. Answers with any additional text are considered incorrect.

    Example of a correct answer:
    B

    Example of an incorrect answer:
    B. Because...

    Answer:
"""


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
    response_key: ClassVar[dict[int, str]] = {0: "A", 1: "B", 2: "C", 3: "D"}
    prompt_template: str = Field(DEFAULT_MMLU_PROMPT)

    def _get_query_from_example(self, example: dict[str, Any]) -> str:
        choices = example["choices"]
        return self.prompt_template.format(
            question=example["question"],
            A=choices[0],
            B=choices[1],
            C=choices[2],
            D=choices[3],
        )

    def _get_response_from_example(self, example: dict[str, Any]) -> str:
        return self.response_key[example["answer"]]

    def _get_context_from_example(self, example: dict[str, Any]) -> str | None:
        return None

    @model_validator(mode="before")
    @classmethod
    def _validate_extra_installed(cls, data: Any) -> Any:
        """Validate that huggingface-evals dependencies are installed."""
        check_huggingface_evals_installed(cls.__name__)
        return data
