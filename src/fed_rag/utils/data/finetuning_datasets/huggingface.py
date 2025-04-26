"""HuggingFace RAG Finetuning Dataset"""

from typing_extensions import Self

from fed_rag.exceptions.common import MissingExtraError

# check if huggingface extra was installed
try:
    from datasets import Dataset
except ModuleNotFoundError:
    msg = (
        "`HuggingFaceRAGFinetuningDataset` requires the `huggingface` extra to be installed. "
        "To fix please run `pip install fed-rag[huggingface]`."
    )
    raise MissingExtraError(msg)


class HuggingFaceRAGFinetuningDataset(Dataset):
    """Thin wrapper over ~datasets.Dataset."""

    @classmethod
    def from_inputs(
        cls,
        input_ids: list[list[int]],
        target_ids: list[list[int]],
        attention_mask: list[list[int]],
    ) -> Self:
        return cls.from_dict(  # type: ignore[no-any-return]
            {
                "input_ids": input_ids,
                "target_ids": target_ids,
                "attention_mask": attention_mask,
            }
        )
