"""Base Benchmark and Benchmarker"""

from abc import ABC, abstractmethod
from typing import Any, Generator, Iterator, Sequence

from pydantic import BaseModel, ConfigDict, PrivateAttr, model_validator

from fed_rag.data_structures.evals import BenchmarkExample
from fed_rag.exceptions import BenchmarkGetExamplesError, BenchmarkParseError


class BaseBenchmark(BaseModel, ABC):
    """Base Benchmark."""

    _examples: Sequence[BenchmarkExample] = PrivateAttr()

    model_config = ConfigDict(arbitrary_types_allowed=True)

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
        try:
            self._examples = self._get_examples()
        except BenchmarkParseError as e:
            raise BenchmarkGetExamplesError(
                f"Failed to parse examples: {str(e)}"
            ) from e
        except Exception as e:
            raise (
                BenchmarkGetExamplesError(f"Failed to get examples: {str(e)}")
            ) from e
        return self

    # abstractmethods
    @abstractmethod
    def _get_examples(self, **kwargs: Any) -> Sequence[BenchmarkExample]:
        """Method to get examples."""

    @abstractmethod
    def as_stream(self) -> Generator[BenchmarkExample, None, None]:
        """Produce a stream of `BenchmarkExamples`."""

    @property
    @abstractmethod
    def num_examples(self) -> int:
        """Number of examples in the benchmark.

        NOTE: if streaming, `_examples` is likely set to an empty list. Thus,
        we leave this implementation for the subclasses.
        """
