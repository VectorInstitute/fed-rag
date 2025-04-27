"""Knowledge Store Utils."""

import json
import logging
from pathlib import Path

from fed_rag.exceptions import KnowledgeStoreNotFoundError

# fed_rag
from fed_rag.knowledge_stores.in_memory import InMemoryKnowledgeStore
from fed_rag.retrievers.huggingface.hf_sentence_transformer import (
    HFSentenceTransformerRetriever,
)
from fed_rag.types.knowledge_node import KnowledgeNode, NodeType

DATA_DIR = Path(__file__).parents[2].absolute() / "data"
ATLAS_DIR = DATA_DIR / "atlas" / "enwiki-dec2021"

ks_logger = logging.getLogger("ra_dit.knowledge_stores")


def knowledge_store_from_retriever(
    retriever: HFSentenceTransformerRetriever,
    name: str,
    persist: bool = False,
    data_path: Path | None = None,
    overwrite: bool = False,
) -> InMemoryKnowledgeStore:
    ks_logger.info(
        f"Creating knowledge store from retriever: name='{name}', persist={persist}, "
        f"data_path={data_path if data_path else 'default'}, overwrite={overwrite}"
    )

    knowledge_store = InMemoryKnowledgeStore(name=name)
    knowledge_store_exists = False

    # try loading from cache
    try:
        knowledge_store.load()
        knowledge_store_exists = True
        ks_logger.info(
            f"Knowledge store successfully loaded from cache: '{name}'."
        )
    except KnowledgeStoreNotFoundError:
        ks_logger.info(
            f"Knowledge store cache not found for '{name}'. Building new store."
        )
        pass

    if not knowledge_store_exists or overwrite:
        if overwrite:
            knowledge_store.clear()
            ks_logger.info("Overwriting the loaded KnowledgeStore.")

        # build the knowledge store by loading chunks from atlast corpus
        data_path = data_path if data_path else ATLAS_DIR
        filename = "sm_sample-text-list-100-sec.jsonl"
        try:
            with open(data_path / filename) as f:
                chunks = [json.loads(line) for line in f]
            ks_logger.info(
                f"Successfully loaded knowledge artifacts from file: {filename}"
            )
            ks_logger.debug(f"Loaded {len(chunks)} chunks from file")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise RuntimeError(f"Failed to load chunks from {filename}: {e}")

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
        ks_logger.info("KnowledgeNode's successfully loaded.")
        ks_logger.debug(
            f"KnowledgeStore has {knowledge_store.count} knowledge nodes"
        )

        # persist
        if persist:
            knowledge_store.persist()
            cached_file = Path(knowledge_store.cache_dir) / (
                knowledge_store.name + ".parquet"
            )
            ks_logger.info(
                f"KnowledgeStore cached to {cached_file.as_posix()}"
            )

    return knowledge_store
