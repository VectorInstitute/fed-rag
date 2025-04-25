"""Dragon Retriever."""

from fed_rag.retrievers.huggingface.hf_sentence_transformer import (
    HFSentenceTransformerRetriever,
)

retriever = HFSentenceTransformerRetriever(
    query_model_name="nthakur/dragon-plus-query-encoder",
    context_model_name="nthakur/dragon-plus-context-encoder",
    load_model_at_init=False,
)

if __name__ == "__main__":
    query = "Where was Marie Curie born?"
    contexts = [
        "Maria Sklodowska, later known as Marie Curie, was born on November 7, 1867.",
        "Born in Paris on 15 May 1859, Pierre Curie was the son of Eugène Curie, a doctor of French Catholic origin from Alsace.",
    ]

    query_embeddings = retriever.encode_query(query)
    context_embeddings = retriever.encode_context(contexts)

    scores = query_embeddings @ context_embeddings.T
    print(scores)
