from typing import Sequence
from unittest.mock import MagicMock

import pytest
import torch

from fed_rag.base.tokenizer import EncodeResult
from fed_rag.types.rag_system import KnowledgeNode, SourceNode
from fed_rag.utils.data import build_finetune_dataset
from fed_rag.utils.data.finetuning_datasets import PyTorchRAGFinetuningDataset
from fed_rag.utils.data.finetuning_datasets.huggingface import (
    HuggingFaceRAGFinetuningDataset,
)


@pytest.fixture()
def mock_examples() -> Sequence[dict]:
    return [
        {
            "query": f"fake query {ix}",
            "answer": f"fake answer {ix}",
            "context": f"fake context {ix}",
        }
        for ix in range(2)
    ]


@pytest.fixture()
def mock_source_nodes() -> list[list[SourceNode]]:
    return [
        [
            SourceNode(
                score=ix / 10,
                node=KnowledgeNode(
                    embedding=[ix, ix, ix],
                    node_type="text",
                    text_content=f"fake text context {ix}",
                ),
            )
            for ix in range(2)
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
            for ix in range(2, 4)
        ],
    ]


def test_build_finetune_dataset_txt_return(
    mock_examples: Sequence[dict], mock_source_nodes: list[list[SourceNode]]
) -> None:
    # arrange
    mock_rag_system = MagicMock()
    mock_retrieve = MagicMock()
    mock_retrieve.side_effect = mock_source_nodes
    mock_tokenizer = MagicMock()
    mock_encode_return: EncodeResult = {
        "attention_mask": None,
        "input_ids": [1, 1, 1],
    }
    mock_tokenizer.encode.return_value = mock_encode_return
    mock_rag_system.retrieve = mock_retrieve
    mock_rag_system.generator.tokenizer = mock_tokenizer

    # act
    result = build_finetune_dataset(
        rag_system=mock_rag_system,
        examples=mock_examples,
        eos_token_id=42,
        return_dataset="txt",
    )

    # assert
    assert result == [
        "fake query 0 fake text context 0 fake answer 0",
        "fake query 0 fake text context 1 fake answer 0",
        "fake query 1 fake text context 2 fake answer 1",
        "fake query 1 fake text context 3 fake answer 1",
    ]


def test_build_finetune_dataset_pt_return(
    mock_examples: Sequence[dict], mock_source_nodes: list[list[SourceNode]]
) -> None:
    # arrange
    mock_rag_system = MagicMock()
    mock_retrieve = MagicMock()
    mock_retrieve.side_effect = mock_source_nodes
    mock_tokenizer = MagicMock()
    mock_encode_return: EncodeResult = {
        "attention_mask": None,
        "input_ids": [1, 1, 1],
    }
    mock_tokenizer.encode.return_value = mock_encode_return
    mock_rag_system.retrieve = mock_retrieve
    mock_rag_system.generator.tokenizer = mock_tokenizer

    # act
    result: PyTorchRAGFinetuningDataset = build_finetune_dataset(
        rag_system=mock_rag_system,
        examples=mock_examples,
        eos_token_id=42,
        return_dataset="pt",
    )

    # assert
    assert isinstance(result, PyTorchRAGFinetuningDataset)
    assert len(result) == 4
    assert isinstance(result.input_ids[0], torch.Tensor)
    assert isinstance(result.target_ids[0], torch.Tensor)


def test_build_finetune_dataset_hf_return(
    mock_examples: Sequence[dict], mock_source_nodes: list[list[SourceNode]]
) -> None:
    # arrange
    mock_rag_system = MagicMock()
    mock_retrieve = MagicMock()
    mock_retrieve.side_effect = mock_source_nodes
    mock_tokenizer = MagicMock()
    mock_encode_return: EncodeResult = {
        "attention_mask": None,
        "input_ids": [1, 1, 1],
    }
    mock_tokenizer.encode.return_value = mock_encode_return
    mock_rag_system.retrieve = mock_retrieve
    mock_rag_system.generator.tokenizer = mock_tokenizer

    # act
    result: HuggingFaceRAGFinetuningDataset = build_finetune_dataset(
        rag_system=mock_rag_system,
        examples=mock_examples,
        eos_token_id=42,
        return_dataset="hf",
    )

    # assert
    assert isinstance(result, HuggingFaceRAGFinetuningDataset)
    assert len(result) == 4
    assert result.column_names == ["input_ids", "target_ids", "attention_mask"]
    assert result[:2] == {
        "input_ids": [
            [1, 1, 1],
            [
                1,
                1,
                1,
            ],
        ],
        "target_ids": [[1, 1, 42], [1, 1, 42]],
        "attention_mask": [None, None],
    }


def test_build_finetune_dataset_invalid_return_raises_error(
    mock_examples: Sequence[dict], mock_source_nodes: list[list[SourceNode]]
) -> None:
    # arrange
    mock_rag_system = MagicMock()
    mock_retrieve = MagicMock()
    mock_retrieve.side_effect = mock_source_nodes
    mock_tokenizer = MagicMock()
    mock_encode_return: EncodeResult = {
        "attention_mask": None,
        "input_ids": [1, 1, 1],
    }
    mock_tokenizer.encode.return_value = mock_encode_return
    mock_rag_system.retrieve = mock_retrieve
    mock_rag_system.generator.tokenizer = mock_tokenizer

    with pytest.raises(ValueError, match="Invalid `return_type` specified."):
        build_finetune_dataset(
            rag_system=mock_rag_system,
            examples=mock_examples,
            eos_token_id=42,
            return_dataset="invalid_return",
        )
