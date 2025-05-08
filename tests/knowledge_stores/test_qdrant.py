import re
import sys
from contextlib import nullcontext as does_not_raise
from unittest.mock import MagicMock, patch

import pytest

from fed_rag.exceptions import (
    InvalidDistanceError,
    KnowledgeStoreError,
    KnowledgeStoreNotFoundError,
    LoadNodeError,
    MissingExtraError,
)
from fed_rag.knowledge_stores.qdrant.sync import (
    QdrantKnowledgeStore,
    _convert_knowledge_node_to_qdrant_point,
    _convert_scored_point_to_knowledge_node_and_score_tuple,
)
from fed_rag.types.knowledge_node import KnowledgeNode


def test_init() -> None:
    knowledge_store = QdrantKnowledgeStore(
        collection_name="test collection",
    )

    assert isinstance(knowledge_store, QdrantKnowledgeStore)
    assert knowledge_store._client is None


def test_init_raises_error_if_qdrant_extra_is_missing_parent_import() -> None:
    modules = {"qdrant_client": None}
    module_to_import = "fed_rag.knowledge_stores.qdrant"

    if module_to_import in sys.modules:
        original_module = sys.modules.pop(module_to_import)

    with patch.dict("sys.modules", modules):
        msg = (
            "Qdrant knowledge stores require the qdrant-client to be installed. "
            "To fix please run `pip install fed-rag[qdrant]`."
        )
        with pytest.raises(
            MissingExtraError,
            match=re.escape(msg),
        ):
            from fed_rag.knowledge_stores.qdrant import QdrantKnowledgeStore

            QdrantKnowledgeStore()

    # restore module so to not affect other tests
    sys.modules[module_to_import] = original_module


def test_init_raises_error_if_qdrant_extra_is_missing() -> None:
    modules = {"qdrant_client": None}
    module_to_import = "fed_rag.knowledge_stores.qdrant.sync"

    if module_to_import in sys.modules:
        original_module = sys.modules.pop(module_to_import)

    with patch.dict("sys.modules", modules):
        msg = (
            "Qdrant knowledge stores require the qdrant-client to be installed. "
            "To fix please run `pip install fed-rag[qdrant]`."
        )
        with pytest.raises(
            MissingExtraError,
            match=re.escape(msg),
        ):
            from fed_rag.knowledge_stores.qdrant.sync import (
                QdrantKnowledgeStore,
            )

            QdrantKnowledgeStore()

    # restore module so to not affect other tests
    sys.modules[module_to_import] = original_module


@patch("qdrant_client.QdrantClient")
def test_get_qdrant_client(mock_qdrant_client_class: MagicMock) -> None:
    knowledge_store = QdrantKnowledgeStore(
        collection_name="test collection",
    )

    # act
    knowledge_store.client

    mock_qdrant_client_class.assert_called_once_with(
        url="http://localhost:6334", api_key=None, prefer_grpc=True
    )


@patch("qdrant_client.QdrantClient")
def test_get_qdrant_client_ssl(mock_qdrant_client_class: MagicMock) -> None:
    knowledge_store = QdrantKnowledgeStore(
        collection_name="test collection", ssl=True
    )

    # act
    knowledge_store.client

    mock_qdrant_client_class.assert_called_once_with(
        url="https://localhost:6334", api_key=None, prefer_grpc=True
    )


@patch("qdrant_client.QdrantClient")
def test_load_node(mock_qdrant_client_class: MagicMock) -> None:
    mock_client = MagicMock()
    mock_qdrant_client_class.return_value = mock_client
    knowledge_store = QdrantKnowledgeStore(
        collection_name="test collection",
    )
    node = KnowledgeNode(
        node_id="1",
        embedding=[1, 1, 1],
        node_type="text",
        text_content="mock node",
    )

    # act
    knowledge_store.load_node(node)

    mock_client.collection_exists.assert_called_once_with("test collection")
    mock_client.upsert.assert_called_once_with(
        collection_name="test collection",
        points=[_convert_knowledge_node_to_qdrant_point(node)],
    )


@patch("qdrant_client.QdrantClient")
def test_load_node_raises_error(mock_qdrant_client_class: MagicMock) -> None:
    mock_client = MagicMock()
    mock_qdrant_client_class.return_value = mock_client
    knowledge_store = QdrantKnowledgeStore(
        collection_name="test collection",
    )
    node = KnowledgeNode(
        node_id="1",
        embedding=[1, 1, 1],
        node_type="text",
        text_content="mock node",
    )

    mock_client.upsert.side_effect = RuntimeError("mock error from qdrant")

    with pytest.raises(
        LoadNodeError,
        match="Failed to load node 1 into collection 'test collection': mock error from qdrant",
    ):
        knowledge_store.load_node(node)


@patch("qdrant_client.QdrantClient")
def test_load_nodes(mock_qdrant_client_class: MagicMock) -> None:
    mock_client = MagicMock()
    mock_qdrant_client_class.return_value = mock_client
    knowledge_store = QdrantKnowledgeStore(
        collection_name="test collection",
    )
    nodes = [
        KnowledgeNode(
            node_id="1",
            embedding=[1, 1, 1],
            node_type="text",
            text_content="mock node",
        ),
        KnowledgeNode(
            node_id="2",
            embedding=[2, 2, 2],
            node_type="text",
            text_content="mock node",
        ),
    ]

    # act
    knowledge_store.load_nodes(nodes)

    mock_client.collection_exists.assert_called_once_with("test collection")
    mock_client.upload_points.assert_called_once_with(
        collection_name="test collection",
        points=[_convert_knowledge_node_to_qdrant_point(n) for n in nodes],
    )

    with does_not_raise():
        knowledge_store.load_nodes([])  # a no-op


@patch("qdrant_client.QdrantClient")
def test_load_nodes_raises_error(mock_qdrant_client_class: MagicMock) -> None:
    mock_client = MagicMock()
    mock_qdrant_client_class.return_value = mock_client
    knowledge_store = QdrantKnowledgeStore(
        collection_name="test collection",
    )
    nodes = [
        KnowledgeNode(
            node_id="1",
            embedding=[1, 1, 1],
            node_type="text",
            text_content="mock node",
        ),
        KnowledgeNode(
            node_id="2",
            embedding=[2, 2, 2],
            node_type="text",
            text_content="mock node",
        ),
    ]

    mock_client.upload_points.side_effect = RuntimeError(
        "mock error from qdrant"
    )

    with pytest.raises(
        LoadNodeError,
        match="Loading nodes into collection 'test collection' failed: mock error from qdrant",
    ):
        knowledge_store.load_nodes(nodes)


@patch("qdrant_client.QdrantClient")
def test_private_ensure_collection_exists(
    mock_qdrant_client_class: MagicMock,
) -> None:
    mock_client = MagicMock()
    mock_qdrant_client_class.return_value = mock_client
    knowledge_store = QdrantKnowledgeStore(
        collection_name="test collection",
    )
    mock_client.collection_exists.return_value = True

    with does_not_raise():
        knowledge_store._ensure_collection_exists()


@patch("qdrant_client.QdrantClient")
def test_private_ensure_collection_exists_raises_not_found(
    mock_qdrant_client_class: MagicMock,
) -> None:
    mock_client = MagicMock()
    mock_qdrant_client_class.return_value = mock_client
    knowledge_store = QdrantKnowledgeStore(
        collection_name="test collection",
    )
    mock_client.collection_exists.return_value = False

    with pytest.raises(
        KnowledgeStoreNotFoundError,
        match="Collection 'test collection' does not exist.",
    ):
        knowledge_store._ensure_collection_exists()


@patch("qdrant_client.QdrantClient")
def test_private_create_collection(
    mock_qdrant_client_class: MagicMock,
) -> None:
    from qdrant_client.models import Distance, VectorParams

    mock_client = MagicMock()
    mock_qdrant_client_class.return_value = mock_client
    knowledge_store = QdrantKnowledgeStore(
        collection_name="test collection",
    )
    distance = Distance(knowledge_store.collection_distance)

    # act
    knowledge_store._create_collection(
        collection_name="test collection", vector_size=100, distance=distance
    )

    mock_client.create_collection.assert_called_once_with(
        collection_name="test collection",
        vectors_config=VectorParams(size=100, distance=distance),
    )


@patch("qdrant_client.QdrantClient")
def test_private_create_collection_raises_invalid_distance_error(
    mock_qdrant_client_class: MagicMock,
) -> None:
    mock_client = MagicMock()
    mock_qdrant_client_class.return_value = mock_client
    knowledge_store = QdrantKnowledgeStore(
        collection_name="test collection",
    )

    # act
    with pytest.raises(InvalidDistanceError):
        knowledge_store._create_collection(
            collection_name="test collection",
            vector_size=100,
            distance="invalid distance",
        )


@patch.object(QdrantKnowledgeStore, "_ensure_collection_exists")
@patch("qdrant_client.QdrantClient")
def test_retrieve(
    mock_qdrant_client_class: MagicMock,
    mock_ensure_collection_exists: MagicMock,
) -> None:
    from qdrant_client.conversions.common_types import QueryResponse
    from qdrant_client.http.models import ScoredPoint

    mock_client = MagicMock()
    mock_qdrant_client_class.return_value = mock_client
    knowledge_store = QdrantKnowledgeStore(
        collection_name="test collection",
    )

    test_node = KnowledgeNode(
        node_id="1",
        embedding=[1, 1, 1],
        node_type="text",
        text_content="mock node",
    )
    test_pt = ScoredPoint(
        id="1", score=0.42, version=1, payload=test_node.model_dump()
    )
    test_query_response = QueryResponse(points=[test_pt])
    mock_client.query_points.return_value = test_query_response

    # act
    retrieval_res = knowledge_store.retrieve(query_emb=[1, 1, 1], top_k=5)

    # assert
    expected = [
        _convert_scored_point_to_knowledge_node_and_score_tuple(test_pt)
    ]
    assert expected == retrieval_res
    mock_ensure_collection_exists.assert_called_once()


@patch.object(QdrantKnowledgeStore, "_ensure_collection_exists")
@patch("qdrant_client.QdrantClient")
def test_retrieve_raises_error(
    mock_qdrant_client_class: MagicMock,
    mock_ensure_collection_exists: MagicMock,
) -> None:
    mock_client = MagicMock()
    mock_qdrant_client_class.return_value = mock_client
    knowledge_store = QdrantKnowledgeStore(
        collection_name="test collection",
    )

    mock_client.query_points.side_effect = RuntimeError("mock qdrant error")

    # act
    with pytest.raises(
        KnowledgeStoreError,
        match="Failed to retrieve from collection 'test collection': mock qdrant error",
    ):
        knowledge_store.retrieve(query_emb=[1, 1, 1], top_k=5)

    mock_ensure_collection_exists.assert_called_once()


def test_persist_raises_error() -> None:
    knowledge_store = QdrantKnowledgeStore(
        collection_name="test collection",
    )

    with pytest.raises(
        NotImplementedError,
        match=re.escape(
            "`persist()` is not available in QdrantKnowledgeStore."
        ),
    ):
        knowledge_store.persist()


def test_load_raises_error() -> None:
    knowledge_store = QdrantKnowledgeStore(
        collection_name="test collection",
    )

    msg = (
        "`load()` is not available in QdrantKnowledgeStore. "
        "Data is automatically persisted and loaded from the Qdrant server."
    )
    with pytest.raises(NotImplementedError, match=re.escape(msg)):
        knowledge_store.load()


@patch.object(QdrantKnowledgeStore, "_ensure_collection_exists")
@patch("qdrant_client.QdrantClient")
def test_delete_node(
    mock_qdrant_client_class: MagicMock,
    mock_ensure_collection_exists: MagicMock,
) -> None:
    from qdrant_client.http.models import (
        FieldCondition,
        Filter,
        MatchValue,
        UpdateResult,
        UpdateStatus,
    )

    mock_client = MagicMock()
    mock_qdrant_client_class.return_value = mock_client
    knowledge_store = QdrantKnowledgeStore(
        collection_name="test collection",
    )

    test_update_result = UpdateResult(status=UpdateStatus.COMPLETED)

    mock_client.delete.return_value = test_update_result

    # act
    knowledge_store.delete_node(node_id="1")

    # assert
    mock_client.delete.assert_called_once_with(
        collection_name="test collection",
        points_selector=Filter(
            must=[FieldCondition(key="node_id", match=MatchValue(value="1"))]
        ),
    )
    mock_ensure_collection_exists.assert_called_once()
