"""MMLU benchmark"""

from typing import Any, Sequence

from pydantic import model_validator

from fed_rag.base.evals.benchmark import BaseBenchmark, BenchmarkExample

from .mixin import HuggingFaceBenchmarkMixin
from .utils import check_huggingface_evals_installed


class HuggingFaceMMLU(HuggingFaceBenchmarkMixin, BaseBenchmark):
    """HuggingFace MMLU Benchmark."""

    dataset_name = "cais/mmlu"

    def _get_examples(self, **kwargs: Any) -> Sequence[BenchmarkExample]:
        raise NotImplementedError

    @model_validator(mode="before")
    @classmethod
    def _validate_extra_installed(cls, data: Any) -> Any:
        """Validate that huggingface-evals dependencies are installed."""
        check_huggingface_evals_installed(cls.__name__)
        return data
