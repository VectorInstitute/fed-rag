"""HuggingFace RAG Finetuning Dataset"""

import torch
from typing_extensions import Self

# check if huggingface extra was installed
try:
    from datasets import Dataset
except ModuleNotFoundError:
    msg = (
        "`HuggingfaceRAGFinetuningDataset` requires the `huggingface` extra to be installed. "
        "To fix please run `pip install fed-rag[huggingface]`."
    )
    raise ValueError(msg)


class HuggingfaceRAGFinetuningDataset(Dataset):
    """Thin wrapper over ~datasets.Dataset."""

    @classmethod
    def from_inputs(
        cls, input_ids: torch.Tensor, target_ids: torch.Tensor
    ) -> Self:
        return cls.from_dict(  # type: ignore[no-any-return]
            {"input_ids": input_ids, "target_ids": target_ids}
        )
