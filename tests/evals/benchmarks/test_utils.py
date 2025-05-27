import tempfile

import pytest

from fed_rag.data_structures import (
    BenchmarkEvaluatedExample,
    BenchmarkExample,
    KnowledgeNode,
    RAGResponse,
    SourceNode,
)
from fed_rag.evals.utils import load_evaluations
from fed_rag.exceptions import EvaluationsFileNotFoundError


@pytest.fixture()
def evaluations() -> list[BenchmarkEvaluatedExample]:
    return [
        BenchmarkEvaluatedExample(
            score=0.42,
            example=BenchmarkExample(
                query="mock query", response="mock response"
            ),
            rag_response=RAGResponse(
                response="mock rag reponse",
                source_nodes=[
                    SourceNode(
                        score=0.1,
                        node=KnowledgeNode(
                            embedding=[1, 2, 3],  # embeddings not persisted
                            node_type="text",
                            text_content="fake content",
                        ),
                    ),
                ],
            ),
        )
    ]


def test_load_evaluations(
    evaluations: list[BenchmarkEvaluatedExample],
) -> None:
    # arrange
    with tempfile.NamedTemporaryFile() as temp:
        with open(temp.name, "w") as f:
            for ev in evaluations:
                f.write(ev.model_dump_json_without_embeddings() + "\n")
                f.flush()

        # act
        loaded_evals = load_evaluations(temp.name)

        # assert
        assert loaded_evals[0].score == evaluations[0].score
        assert loaded_evals[0].example == evaluations[0].example
        assert (
            loaded_evals[0].rag_response.response
            == evaluations[0].rag_response.response
        )
        assert (
            loaded_evals[0].rag_response.source_nodes[0].node.embedding is None
        )


def test_load_evaluations_raises_error_file_not_found() -> None:
    with pytest.raises(
        EvaluationsFileNotFoundError,
        match="__test.jsonl",
    ):
        load_evaluations("__test.jsonl")
