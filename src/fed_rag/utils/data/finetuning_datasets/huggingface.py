"""HuggingFace RAG Finetuning Dataset"""

import torch
from typing_extensions import Self

# check if huggingface extra was installed
try:
    from datasets import Dataset

    _has_huggingface = True
except ModuleNotFoundError:
    _has_huggingface = False


class HuggingfaceRAGFinetuningDataset(Dataset):
    """Thin wrapper over ~datasets.Dataset."""

    @classmethod
    def from_inputs(
        cls, input_ids: torch.Tensor, target_ids: torch.Tensor
    ) -> Self:
        if not _has_huggingface:
            msg = (
                "HuggingFace finetuning datasets requires the `huggingface` extra to be installed. "
                "To fix please run `pip install fed-rag[huggingface]`."
            )
            raise ValueError(msg)
        return cls.from_dict(  # type: ignore[no-any-return]
            {"input_ids": input_ids, "target_ids": target_ids}
        )
