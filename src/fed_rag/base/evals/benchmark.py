"""Base Benchmark and Benchmarker"""

from abc import ABC, abstractmethod
from typing import Any, Generator, Iterator, Sequence

from pydantic import BaseModel, ConfigDict, PrivateAttr, model_validator

from fed_rag.data_structures.evals import BenchmarkExample
from fed_rag.exceptions import BenchmarkGetExamplesError, BenchmarkParseError


class BaseBenchmark(BaseModel, ABC):
    """Base class for implementing benchmarks.

    This abstract class defines the interface for benchmark datasets,
    providing methods to access examples, iterate over them, and stream
    them lazily. Subclasses must implement how examples are retrieved
    and how many examples exist.
    """

    _examples: Sequence[BenchmarkExample] = PrivateAttr()

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __getitem__(self, index: int) -> BenchmarkExample:
        """Return a benchmark example at the specified index.

        Args:
            index (int): The position of the example in the sequence.

        Returns:
            BenchmarkExample: The benchmark example at the given index.
        """
        return self._examples.__getitem__(index)

    def __len__(self) -> int:
        """Return the number of loaded examples.

        Returns:
            int: The number of examples currently loaded into memory.
        """
        return self._examples.__len__()

    def as_iterator(self) -> Iterator[BenchmarkExample]:
        """Return an iterator over the loaded examples.

        Note:
            This uses the in-memory examples. For large datasets that
            cannot fit into memory, use :meth:`as_stream` instead.

        Returns:
            Iterator[BenchmarkExample]: Iterator over benchmark examples.
        """
        return self._examples.__iter__()

    @model_validator(mode="after")
    def set_examples(self) -> "BaseBenchmark":
        """Populate the benchmark with examples after initialization.

        Returns:
            BaseBenchmark: The instance with examples set.

        Raises:
            BenchmarkGetExamplesError: If retrieving or parsing examples fails.
        """
        try:
            self._examples = self._get_examples()
        except BenchmarkParseError as e:
            raise BenchmarkGetExamplesError(
                f"Failed to parse examples: {str(e)}"
            ) from e
        except Exception as e:
            raise BenchmarkGetExamplesError(
                f"Failed to get examples: {str(e)}"
            ) from e
        return self

    @abstractmethod
    def _get_examples(self, **kwargs: Any) -> Sequence[BenchmarkExample]:
        """Fetch and return all benchmark examples.

        Args:
            **kwargs (Any): Optional arguments for retrieving examples.

        Returns:
            Sequence[BenchmarkExample]: A sequence of benchmark examples.

        Raises:
            BenchmarkParseError: If parsing examples fails.
        """
        ...

    @abstractmethod
    def as_stream(self) -> Generator[BenchmarkExample, None, None]:
        """Stream benchmark examples one by one.

        This method is useful for very large datasets that cannot be
        stored entirely in memory.

        Yields:
            Generator[BenchmarkExample, None, None]: Benchmark examples.
        """
        ...

    @property
    @abstractmethod
    def num_examples(self) -> int:
        """Return the total number of examples in the benchmark.

        Note:
            If streaming is used, `_examples` may be an empty list. In such
            cases, subclasses should implement their own logic for counting.

        Returns:
            int: Total number of examples.
        """
        ...
