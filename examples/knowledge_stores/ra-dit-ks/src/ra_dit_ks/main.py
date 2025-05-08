"""Knowledge Store Utils."""

import itertools
import json
import logging
import os
import time
from pathlib import Path
from typing import Generator

from dotenv import load_dotenv

# fed_rag
from fed_rag.exceptions import KnowledgeStoreNotFoundError
from fed_rag.knowledge_stores.qdrant import QdrantKnowledgeStore
from fed_rag.retrievers.huggingface.hf_sentence_transformer import (
    HFSentenceTransformerRetriever,
)
from fed_rag.types.knowledge_node import KnowledgeNode, NodeType

DATA_DIR = Path(__file__).parents[2].absolute() / "data"
ATLAS_DIR = DATA_DIR / "atlas" / "enwiki-dec2021"

ks_logger = logging.getLogger("ra_dit_ks.main")


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
            model_name=model_name,
            load_model_at_init=False,
            load_model_kwargs={"device_map": "auto"},
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
    env_file_path: str | None = None,
) -> QdrantKnowledgeStore:
    collection_name = collection_name or (
        retriever.model_name or retriever.context_model_name
    )
    collection_name = collection_name.replace("/", ".")
    ks_logger.info(
        f"Creating knowledge store from retriever: collection_name='{collection_name}', "
        f"data_path={data_path if data_path else 'default'}, clear_first={clear_first} "
        f"batch_size={batch_size} and num_parallel_load={num_parallel_load}."
    )
    sentence_transformer = retriever.encoder or retriever.context_encoder
    ks_logger.debug(f"Retriever on device: {sentence_transformer.device}")

    knowledge_store_kwargs = {
        "collection_name": collection_name,
        "load_node_kwargs": {"parallel": num_parallel_load},
    }
    if env_file_path:
        load_dotenv(dotenv_path=env_file_path)
        host = os.environ.get("QDRANT_HOST")
        api_key = os.environ.get("QDRANT_API_KEY")
        knowledge_store_kwargs.update(host=host, api_key=api_key, ssl=True)

    knowledge_store = QdrantKnowledgeStore(**knowledge_store_kwargs)

    if clear_first:
        try:
            knowledge_store.clear()
            ks_logger.info("Knowledge store has been successfully cleared.")
        except KnowledgeStoreNotFoundError:
            ks_logger.info("Knowledge store nonexistent. Nothing to clear.")

    # build the knowledge store by loading chunks from atlast corpus
    data_path = data_path if data_path else ATLAS_DIR
    filename = "text-list-100-sec.jsonl"

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
        node_creation_start_time = time.time()
        nodes = []
        texts = []
        for c in chunks:
            text = c.pop("text")
            title = c.pop("title")
            section = c.pop("section")
            context_text = f"title: {title}\nsection: {section}\ntext: {text}"
            texts.append(context_text)

        # batch encode
        batch_embeddings = retriever.encode_context(texts)

        for jx, c in enumerate(chunks):
            node = KnowledgeNode(
                embedding=batch_embeddings[jx].tolist(),
                node_type=NodeType.TEXT,
                text_content=texts[jx],
                metadata=c,
            )
            nodes.append(node)
        ks_logger.info(
            f"KnowledgeNode's successfully created in: {time.time() - node_creation_start_time:.2f} seconds"
        )
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
    num_parallel_load: int = 1,
    env_file_path: str | None = None,
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
        num_parallel_load=num_parallel_load,
        env_file_path=env_file_path,
    )
    ks_logger.info(
        f"Knowledge store creation took {time.time() - start_time:.2f} seconds"
    )

    return (knowledge_store.collection_name, knowledge_store.count)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
