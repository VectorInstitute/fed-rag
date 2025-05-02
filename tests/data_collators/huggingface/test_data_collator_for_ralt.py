import re
import sys
from typing import Any, Sequence
from unittest.mock import MagicMock, patch

import pytest
import torch
from pytest import MonkeyPatch
from torch.testing import assert_close

from fed_rag.base.tokenizer import EncodeResult
from fed_rag.data_collators.huggingface.ralt import (
    DEFAULT_EXAMPLE_TEMPLATE,
    DataCollatorForLanguageModeling,
    DataCollatorForRALT,
)
from fed_rag.exceptions import (
    DataCollatorError,
    FedRAGError,
    MissingExtraError,
)
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
    module_to_import = "fed_rag.data_collators.huggingface.ralt"
    original_module = sys.modules.pop(module_to_import, None)

    with patch.dict("sys.modules", modules):
        msg = (
            "`DataCollatorForRALT` requires `huggingface` extra to be installed. "
            "To fix please run `pip install fed-rag[huggingface]`."
        )
        with pytest.raises(
            MissingExtraError,
            match=re.escape(msg),
        ):
            from fed_rag.data_collators.huggingface.ralt import (
                DataCollatorForRALT,
            )

            DataCollatorForRALT(
                rag_system=mock_rag_system,
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
            from fed_rag.data_collators.huggingface import DataCollatorForRALT

            DataCollatorForRALT(
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
        from fed_rag.data_collators.huggingface import DataCollatorForRALT

        DataCollatorForRALT(rag_system=mock_rag_system)


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
        from fed_rag.data_collators.huggingface import DataCollatorForRALT

        DataCollatorForRALT(rag_system=mock_rag_system, prompt_template="")


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
        collator = DataCollatorForRALT(
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

    collator = DataCollatorForRALT(
        rag_system=mock_rag_system,
    )

    assert collator.rag_system == mock_rag_system
    assert collator.example_template == DEFAULT_EXAMPLE_TEMPLATE
    assert collator.default_return_tensors == "pt"


@pytest.fixture()
def mock_examples() -> Sequence[dict]:
    return [
        {
            "query": f"fake query {ix}",
            "response": f"fake response {ix}",
            "context": f"fake context {ix}",
        }
        for ix in range(2)
    ]


@patch.object(
    DataCollatorForLanguageModeling,
    "torch_call",
)
@patch.object(RAGSystem, "retrieve")
def test_lsr_collator_with_mocks(
    mock_retrieve: MagicMock,
    mock_torch_call: MagicMock,
    mock_rag_system: RAGSystem,
    mock_examples: Sequence[dict],
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

    # arrange mocks
    mock_tokenizer = MagicMock()
    mock_encode_return: EncodeResult = {
        "attention_mask": None,
        "input_ids": [1, 1, 1],
    }
    mock_tokenizer.encode.return_value = mock_encode_return
    mock_tokenizer.unwrapped.all_special_ids.__getitem__.return_value = 42
    rag_system.generator.tokenizer = mock_tokenizer

    # mock top-k = 2
    mock_top_k_val = 2
    mock_retrieve.side_effect = [
        [
            SourceNode(
                score=ix / 10,
                node=KnowledgeNode(
                    embedding=[ix, ix, ix],
                    node_type="text",
                    text_content=f"fake text context {ix}",
                ),
            )
            for ix in range(mock_top_k_val)
        ],
        [
            SourceNode(
                score=ix / 10,
                node=KnowledgeNode(
                    embedding=[ix, ix, ix],
                    node_type="text",
                    text_content=f"fake text context {ix}",
                ),
            )
            for ix in range(mock_top_k_val, mock_top_k_val * 2)
        ],
    ]

    def _mock_torch_call(features: Any) -> dict[str, torch.Tensor]:
        retval = {
            k: torch.stack([torch.tensor(el) for el in v], dim=0)
            for k, v in features.items()
            if k in ["input_ids", "target_ids"]
        }
        retval["labels"] = retval["target_ids"]
        del retval["target_ids"]
        return retval

    # Create a mock instance to be returned when the class is instantiated
    # mock_torch_call = MagicMock()
    mock_torch_call.side_effect = _mock_torch_call

    # arrange collator
    collator = DataCollatorForRALT(
        rag_system=mock_rag_system,
        example_template="{query} {context} {response}",
    )

    # act
    collated_batch = collator(mock_examples)

    expected_num_examples = mock_top_k_val * len(mock_examples)
    assert collated_batch["input_ids"].shape[0] == expected_num_examples
    mock_torch_call.assert_called_once()
    assert_close(
        collated_batch["input_ids"],
        torch.tensor([[1, 1, 1] for _ in range(expected_num_examples)]),
    )
    assert_close(
        collated_batch["labels"],
        torch.tensor([[1, 1, 42] for _ in range(expected_num_examples)]),
    )


@patch.object(RAGSystem, "retrieve")
def test_lsr_collator_raises_data_collator_error_when_missing_eos_token_id(
    mock_retrieve: MagicMock,
    mock_rag_system: RAGSystem,
    mock_examples: Sequence[dict],
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

    # arrange mocks
    mock_tokenizer = MagicMock()
    mock_encode_return: EncodeResult = {
        "attention_mask": None,
        "input_ids": [1, 1, 1],
    }
    mock_tokenizer.encode.return_value = mock_encode_return
    mock_tokenizer.unwrapped.special_tokens_map.get.side_effect = KeyError
    rag_system.generator.tokenizer = mock_tokenizer

    # mock top-k = 2
    mock_top_k_val = 2
    mock_retrieve.side_effect = [
        [
            SourceNode(
                score=ix / 10,
                node=KnowledgeNode(
                    embedding=[ix, ix, ix],
                    node_type="text",
                    text_content=f"fake text context {ix}",
                ),
            )
            for ix in range(mock_top_k_val)
        ],
        [
            SourceNode(
                score=ix / 10,
                node=KnowledgeNode(
                    embedding=[ix, ix, ix],
                    node_type="text",
                    text_content=f"fake text context {ix}",
                ),
            )
            for ix in range(mock_top_k_val, mock_top_k_val * 2)
        ],
    ]

    # arrange collator
    collator = DataCollatorForRALT(
        rag_system=mock_rag_system,
        example_template="{query} {context} {response}",
    )

    # act
    with pytest.raises(
        DataCollatorError, match="Tokenizer doesn't have an `eos_token`."
    ):
        collator(mock_examples)
