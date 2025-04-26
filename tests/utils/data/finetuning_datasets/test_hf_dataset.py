import re
import sys
from unittest.mock import patch

import pytest
import torch
from datasets import Dataset

from fed_rag.exceptions import MissingExtraError
from fed_rag.utils.data.finetuning_datasets.huggingface import (
    HuggingFaceRAGFinetuningDataset,
)


def test_hf_rag_ft_dataset_init(
    input_and_target_ids: tuple[torch.Tensor, torch.Tensor],
) -> None:
    input_ids, target_ids = input_and_target_ids
    rag_ft_dataset = HuggingFaceRAGFinetuningDataset.from_inputs(
        input_ids=input_ids,
        target_ids=target_ids,
        attention_mask=[None] * len(target_ids),
    )

    assert len(rag_ft_dataset) == len(input_ids)
    assert isinstance(rag_ft_dataset, Dataset)
    assert rag_ft_dataset["input_ids"] == [t.tolist() for t in input_ids]
    assert rag_ft_dataset["target_ids"] == [t.tolist() for t in target_ids]
    assert rag_ft_dataset["attention_mask"] == [None, None, None]


def test_hf_rag_ft_dataset_missing_extra_raises_error(
    input_and_target_ids: tuple[torch.Tensor, torch.Tensor],
) -> None:
    input_ids, target_ids = input_and_target_ids
    modules = {"datasets": None}
    module_to_import = "fed_rag.utils.data.finetuning_datasets.huggingface"

    if module_to_import in sys.modules:
        original_module = sys.modules.pop(module_to_import)

    with patch.dict("sys.modules", modules):
        msg = (
            "`HuggingFaceRAGFinetuningDataset` requires the `huggingface` extra to be installed. "
            "To fix please run `pip install fed-rag[huggingface]`."
        )

        with pytest.raises(
            MissingExtraError,
            match=re.escape(msg),
        ):
            from fed_rag.utils.data.finetuning_datasets.huggingface import (
                HuggingFaceRAGFinetuningDataset,
            )

            HuggingFaceRAGFinetuningDataset.from_inputs(
                input_ids=input_ids, target_ids=target_ids
            )

    # restore module so to not affect other tests
    sys.modules[module_to_import] = original_module
