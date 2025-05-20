from typing import Any, Generator

import pytest
from datasets import Dataset, DatasetInfo, IterableDataset, Split


@pytest.fixture
def dummy_dataset() -> Dataset:
    benchmark_info = DatasetInfo(
        dataset_name="test_benchmark",
        description="A toy RAG dataset for testing purposes",
    )
    benchmark = Dataset.from_dict(
        {
            "query": ["a query", "another query", "yet another query"],
            "response": [
                "reponse to a query",
                "another response to another query",
                "yet another response to yet another query",
            ],
            "context": [
                "context for a query",
                "another context for another query",
                "yet another context for yet another query",
            ],
        },
        info=benchmark_info,
        split=Split.TEST,
    )
    return benchmark


@pytest.fixture
def dummy_iterable_dataset() -> IterableDataset:
    def example_gen() -> Generator[dict[str, Any], None, None]:
        yield {
            "query": "a query",
            "response": "response to a query",
            "context": "context for a query",
        }
        yield {
            "query": "another query",
            "response": "another response to another query",
            "context": "another context for another query",
        }
        yield {
            "query": "yet another query",
            "response": "yet another response to yet another query",
            "context": "yet another context for yet another query",
        }

    return IterableDataset.from_generator(example_gen, split=Split.TEST)


@pytest.fixture
def dummy_mmlu() -> Dataset:
    benchmark_info = DatasetInfo(
        dataset_name="cais/mmlu",
    )
    benchmark = Dataset.from_dict(
        {
            "question": [
                "What is the embryological origin of the hyoid bone?"
            ],
            "choices": [
                [
                    "The first pharyngeal arch",
                    "The first and second pharyngeal arches",
                    "The second pharyngeal arch",
                    "The second and third pharyngeal arches",
                ]
            ],
            "answer": [3],
        },
        info=benchmark_info,
        split=Split.TEST,
    )
    return benchmark
