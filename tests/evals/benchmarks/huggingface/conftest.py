import pytest
from datasets import Dataset, DatasetInfo, Split


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
