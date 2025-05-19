"""Base Benchmark and Benchmarker"""

from abc import ABC, abstractmethod
from typing import Any, Iterator, Sequence

from pydantic import BaseModel, ConfigDict, PrivateAttr, model_validator

from fed_rag.data_structures.evals import BenchmarkExample


class BaseBenchmark(BaseModel, ABC):
    """Base Benchmark."""

    _examples: Sequence[BenchmarkExample] = PrivateAttr()

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    def _get_examples(self, **kwargs: Any) -> Sequence[BenchmarkExample]:
        """Method to get examples."""

    # give it a sequence interface for accessing examples more easily
    def __getitem__(self, index: int) -> BenchmarkExample:
        return self._examples.__getitem__(index)

    def __len__(self) -> int:
        return self._examples.__len__()

    # shouldn't override Pydantic BaseModels' __iter__
    def as_iterator(self) -> Iterator[BenchmarkExample]:
        return self._examples.__iter__()

    @model_validator(mode="after")
    def set_examples(self) -> "BaseBenchmark":
        print("setting examples")
        self._examples = self._get_examples()
        return self
