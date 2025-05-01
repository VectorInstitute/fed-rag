import re
import sys
from unittest.mock import MagicMock, _Call, patch

import pytest
import torch
from pytest import MonkeyPatch
from torch.testing import assert_close

from fed_rag.data_collators.huggingface import DataCollatorForLSR
from fed_rag.exceptions import MissingExtraError
from fed_rag.exceptions.core import FedRAGError
from fed_rag.generators.huggingface import HFPeftModelGenerator
from fed_rag.retrievers.huggingface.hf_sentence_transformer import (
    HFSentenceTransformerRetriever,
)
from fed_rag.types.knowledge_node import KnowledgeNode
from fed_rag.types.rag_system import RAGSystem, SourceNode


def test_huggingface_extra_missing(mock_rag_system: RAGSystem) -> None:
    """Test extra is not installed."""

    modules = {
        "transformers": None,
        "transformers.data": None,
        "transformers.data.data_collator": None,
    }
    module_to_import = "fed_rag.data_collators.huggingface.lsr"
    original_module = sys.modules.pop(module_to_import, None)

    with patch.dict("sys.modules", modules):
        msg = (
            "`DataCollatorForLSR` requires `huggingface` extra to be installed. "
            "To fix please run `pip install fed-rag[huggingface]`."
        )
        with pytest.raises(
            MissingExtraError,
            match=re.escape(msg),
        ):
            from fed_rag.data_collators.huggingface.lsr import (
                DataCollatorForLSR,
            )

            DataCollatorForLSR(
                rag_system=mock_rag_system,
                prompt_template="{query} and {context}",
            )

    # restore module so to not affect other tests
    if original_module:
        sys.modules[module_to_import] = original_module


def test_huggingface_extra_missing_from_parent(
    mock_rag_system: RAGSystem,
) -> None:
    """Test extra is not installed."""

    modules = {
        "transformers": None,
        "transformers.data": None,
        "transformers.data.data_collator": None,
    }
    module_to_import = "fed_rag.data_collators.huggingface"
    original_module = sys.modules.pop(module_to_import, None)

    with patch.dict("sys.modules", modules):
        msg = (
            "`fed_rag.data_collators.huggingface` requires `huggingface` extra to be installed. "
            "To fix please run `pip install fed-rag[huggingface]`."
        )
        with pytest.raises(
            MissingExtraError,
            match=re.escape(msg),
        ):
            from fed_rag.data_collators.huggingface import DataCollatorForLSR

            DataCollatorForLSR(
                rag_system=mock_rag_system,
                prompt_template="{query} and {context}",
            )

    # restore module so to not affect other tests
    if original_module:
        sys.modules[module_to_import] = original_module


def test_invalid_rag_system_due_to_generators(
    mock_rag_system: RAGSystem,
) -> None:
    with pytest.raises(
        FedRAGError,
        match="Generator must be HFPretrainedModelGenerator or HFPeftModelGenerator",
    ):
        from fed_rag.data_collators.huggingface import DataCollatorForLSR

        DataCollatorForLSR(rag_system=mock_rag_system, prompt_template="")


def test_invalid_rag_system_due_to_retriever(
    mock_rag_system: RAGSystem,
) -> None:
    generator = HFPeftModelGenerator(
        model_name="fake_name",
        base_model_name="fake_base_name",
        load_model_at_init=False,
    )
    mock_rag_system.generator = generator

    with pytest.raises(
        FedRAGError,
        match="Retriever must be a HFSentenceTransformerRetriever",
    ):
        from fed_rag.data_collators.huggingface import DataCollatorForLSR

        DataCollatorForLSR(rag_system=mock_rag_system, prompt_template="")


def test_invalid_return_tensors(
    mock_rag_system: RAGSystem,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("FEDRAG_SKIP_VALIDATION", "1")
    return_tensors = "af"
    with pytest.raises(
        FedRAGError,
        match=f"Framework '{return_tensors}' not recognized!",
    ):
        collator = DataCollatorForLSR(
            rag_system=mock_rag_system, prompt_template=""
        )
        collator([], return_tensors)


def test_init(
    mock_rag_system: RAGSystem,
) -> None:
    generator = HFPeftModelGenerator(
        model_name="fake_name",
        base_model_name="fake_base_name",
        load_model_at_init=False,
    )
    retriever = HFSentenceTransformerRetriever(
        model_name="fake_name", load_model_at_init=False
    )
    mock_rag_system.generator = generator
    mock_rag_system.retriever = retriever

    collator = DataCollatorForLSR(
        rag_system=mock_rag_system, prompt_template="{query} and {context}"
    )

    assert collator.rag_system == mock_rag_system
    assert collator.prompt_template == "{query} and {context}"
    assert collator.default_return_tensors == "pt"


@patch.object(RAGSystem, "retrieve")
def test_lsr_collator_with_mocks(
    mock_retrieve: MagicMock,
    mock_rag_system: RAGSystem,
    monkeypatch: MonkeyPatch,
) -> None:
    # Set environment variable for the duration of this test only
    monkeypatch.setenv("FEDRAG_SKIP_VALIDATION", "1")

    rag_system = RAGSystem(
        generator=mock_rag_system.generator,
        retriever=mock_rag_system.retriever,
        knowledge_store=mock_rag_system.knowledge_store,
        rag_config=mock_rag_system.rag_config,
    )

    # use mocks
    mock_generator = MagicMock()
    mock_generator.compute_target_sequence_proba.side_effect = [
        torch.tensor(0.01),
        torch.tensor(0.02),
        torch.tensor(0.03),
    ]
    rag_system.generator = mock_generator

    mock_retrieve.return_value = [
        SourceNode(
            score=0.1,
            node=KnowledgeNode(
                embedding=[0.1], node_type="text", text_content="node 1"
            ),
        ),
        SourceNode(
            score=0.2,
            node=KnowledgeNode(
                embedding=[0.2], node_type="text", text_content="node 2"
            ),
        ),
        SourceNode(
            score=0.3,
            node=KnowledgeNode(
                embedding=[0.3], node_type="text", text_content="node 3"
            ),
        ),
    ]

    collator = DataCollatorForLSR(
        rag_system=rag_system, prompt_template="{query} {context}"
    )

    # act
    features = [{"query": "mock query", "response": "mock response"}]
    batch = collator(features)

    assert_close(
        batch["retrieval_scores"], torch.tensor([0.1, 0.2, 0.3]).unsqueeze(0)
    )
    assert_close(
        batch["lm_scores"], torch.tensor([0.01, 0.02, 0.03]).unsqueeze(0)
    )
    mock_retrieve.assert_called_once_with("mock query")
    mock_generator.compute_target_sequence_proba.assert_has_calls(
        [
            _Call(
                (
                    (),
                    {
                        "prompt": "mock query node 1",
                        "target": "\n<response>\nmock response\n</response>\n",
                    },
                )
            ),
            _Call(
                (
                    (),
                    {
                        "prompt": "mock query node 2",
                        "target": "\n<response>\nmock response\n</response>\n",
                    },
                )
            ),
            _Call(
                (
                    (),
                    {
                        "prompt": "mock query node 3",
                        "target": "\n<response>\nmock response\n</response>\n",
                    },
                )
            ),
        ]
    )
