import re
from unittest.mock import patch

import pytest
import torch
from datasets import Dataset

from fed_rag.utils.data.finetuning_datasets import (
    HuggingfaceRAGFinetuningDataset,
)


def test_hf_rag_ft_dataset_init(
    input_and_target_ids: tuple[torch.Tensor, torch.Tensor],
) -> None:
    input_ids, target_ids = input_and_target_ids
    rag_ft_dataset = HuggingfaceRAGFinetuningDataset.from_inputs(
        input_ids=input_ids, target_ids=target_ids
    )

    assert len(rag_ft_dataset) == len(input_ids)
    assert isinstance(rag_ft_dataset, Dataset)
    assert rag_ft_dataset["input_ids"] == [t.tolist() for t in input_ids]
    assert rag_ft_dataset["target_ids"] == [t.tolist() for t in target_ids]


@patch(
    "fed_rag.utils.data.finetuning_datasets.huggingface._has_huggingface",
    False,
)
def test_hf_rag_ft_dataset_missing_extra_raises_error(
    input_and_target_ids: tuple[torch.Tensor, torch.Tensor],
) -> None:
    input_ids, target_ids = input_and_target_ids
    msg = (
        "`HuggingfaceRAGFinetuningDataset` requires the `huggingface` extra to be installed. "
        "To fix please run `pip install fed-rag[huggingface]`."
    )
    with pytest.raises(
        ValueError,
        match=re.escape(msg),
    ):
        from fed_rag.utils.data.finetuning_datasets import (
            HuggingfaceRAGFinetuningDataset,
        )

        HuggingfaceRAGFinetuningDataset.from_inputs(
            input_ids=input_ids, target_ids=target_ids
        )
