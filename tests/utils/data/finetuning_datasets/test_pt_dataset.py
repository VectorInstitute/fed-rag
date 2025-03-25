import torch

from fed_rag.utils.data.finetuning_datasets import PyTorchRAGFinetuningDataset


def test_pt_rag_ft_dataset_init(
    input_and_target_ids: tuple[torch.Tensor, torch.Tensor],
) -> None:
    input_ids, target_ids = input_and_target_ids
    rag_ft_dataset = PyTorchRAGFinetuningDataset(
        input_ids=input_ids, target_ids=target_ids
    )

    assert len(rag_ft_dataset) == len(input_ids)
    assert isinstance(rag_ft_dataset, torch.utils.data.Dataset)
    assert rag_ft_dataset[:] == input_and_target_ids[:]
