from fed_rag.data_structures import (
    BenchmarkEvaluatedExample,
    BenchmarkExample,
    KnowledgeNode,
    RAGResponse,
    SourceNode,
)


def test_model_dump_without_embs() -> None:
    evaluated = BenchmarkEvaluatedExample(
        score=0.42,
        example=BenchmarkExample(query="mock query", response="mock response"),
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

    # act
    json_str = evaluated.model_dump_json_without_embeddings()

    # assert
    loaded_evaluated = BenchmarkEvaluatedExample.model_validate_json(json_str)
    assert loaded_evaluated.rag_response.source_nodes[0].node.embedding is None
