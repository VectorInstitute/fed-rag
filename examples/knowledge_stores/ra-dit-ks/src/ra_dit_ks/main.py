"""Knowledge Store Utils."""

import itertools
import json
import logging
import time
from pathlib import Path
from typing import Generator

# fed_rag
from fed_rag.knowledge_stores.qdrant import QdrantKnowledgeStore
from fed_rag.retrievers.huggingface.hf_sentence_transformer import (
    HFSentenceTransformerRetriever,
)
from fed_rag.types.knowledge_node import KnowledgeNode, NodeType

DATA_DIR = Path(__file__).parents[2].absolute() / "data"
ATLAS_DIR = DATA_DIR / "atlas" / "enwiki-dec2021"

ks_logger = logging.getLogger("ra_dit_ks.main")


retriever = HFSentenceTransformerRetriever(
    query_model_name="nthakur/dragon-plus-query-encoder",
    context_model_name="nthakur/dragon-plus-context-encoder",
    load_model_at_init=False,
)


def get_retriever(
    model_name: str | None,
    query_model_name: str | None = "nthakur/dragon-plus-query-encoder",
    context_model_name: str | None = "nthakur/dragon-plus-context-encoder",
) -> HFSentenceTransformerRetriever:
    if model_name is not None and (
        query_model_name is not None or context_model_name is not None
    ):
        raise RuntimeError(
            "`model_name` cannot be specified if either `query_model_name` or `context_model_name` are also specified."
        )

    if model_name:
        ks_logger.info(f"Getting retriever: model_name='{model_name}'.")
        return HFSentenceTransformerRetriever(
            model_name=model_name, load_model_at_init=False
        )
    else:
        ks_logger.info(
            f"Getting retriever: query_model_name='{query_model_name}' and "
            f"context_model_name='{context_model_name}'."
        )
        return HFSentenceTransformerRetriever(
            query_model_name=query_model_name,
            context_model_name=context_model_name,
            load_model_at_init=False,
        )


def knowledge_store_from_retriever(
    retriever: HFSentenceTransformerRetriever,
    collection_name: str | None,
    data_path: Path | None = None,
    clear_first: bool = False,
    num_parallel_load: int = 1,
    batch_size: int = 1000,
) -> QdrantKnowledgeStore:
    collection_name = collection_name or (
        retriever.model_name or retriever.context_model_name
    )
    collection_name = collection_name.replace("/", ".")
    ks_logger.info(
        f"Creating knowledge store from retriever: collection_name='{collection_name}', "
        f"data_path={data_path if data_path else 'default'}, clear_first={clear_first} "
        f"and batch_size={batch_size}."
    )

    knowledge_store = QdrantKnowledgeStore(
        collection_name=collection_name,
        load_node_kwargs={"parallel": num_parallel_load},
    )

    if clear_first:
        knowledge_store.clear()
        ks_logger.info("Knowledge store has been successfully cleared.")

    # build the knowledge store by loading chunks from atlast corpus
    data_path = data_path if data_path else ATLAS_DIR
    filename = "md_sample-text-list-100-sec.jsonl"

    def batch_stream_file(
        filename: str, batch_size: int
    ) -> Generator[list[str], None, None]:
        """batch stream file"""
        with open(data_path / filename) as f:
            while True:
                batch = [
                    line.strip() for line in itertools.islice(f, batch_size)
                ]

                if not batch:
                    break

                yield batch

    for ix, batch in enumerate(
        batch_stream_file(filename, batch_size=batch_size)
    ):
        try:
            chunks = [json.loads(line) for line in batch]
            ks_logger.info(
                f"Successfully loaded knowledge artifacts from file: {filename} and batch: {ix + 1}"
            )
            ks_logger.debug(f"Loaded {len(chunks)} chunks from file")
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"Failed to load chunks from {filename} and batch {ix + 1}: {e}"
            ) from e

        # create knowledge nodes
        nodes = []
        for c in chunks:
            text = c.pop("text")
            node = KnowledgeNode(
                embedding=retriever.encode_context(text).tolist(),
                node_type=NodeType.TEXT,
                text_content=text,
                metadata=c,
            )
            nodes.append(node)
        ks_logger.info("KnowledgeNode's successfully created.")
        ks_logger.debug(f"Created {len(nodes)} knowledge nodes")

        # load into knowledge_store
        knowledge_store.load_nodes(nodes=nodes)
        ks_logger.info(
            f"KnowledgeNode's successfully loaded {len(nodes)} for batch: {ix + 1}."
        )
        ks_logger.debug(
            f"KnowledgeStore now has a total of {knowledge_store.count} knowledge nodes"
        )

    return knowledge_store


def main(
    model_name: str | None = None,
    query_model_name: str | None = "nthakur/dragon-plus-query-encoder",
    context_model_name: str | None = "nthakur/dragon-plus-context-encoder",
    collection_name: str | None = None,
    data_path: Path | None = None,
    clear_first: bool = False,
    batch_size: int = 1000,
) -> tuple[str, int]:
    # get retriever
    retriever = get_retriever(
        model_name=model_name,
        query_model_name=query_model_name,
        context_model_name=context_model_name,
    )

    # build knowledge store
    start_time = time.time()
    knowledge_store = knowledge_store_from_retriever(
        retriever=retriever,
        collection_name=collection_name,
        data_path=data_path,
        clear_first=clear_first,
        batch_size=batch_size,
    )
    ks_logger.info(
        f"Knowledge store creation took {time.time() - start_time:.2f} seconds"
    )

    return (knowledge_store.collection_name, knowledge_store.count)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
