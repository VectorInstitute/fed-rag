"""Utils module."""

from enum import Enum

from pydantic import BaseModel, PrivateAttr

from fed_rag.generators.hf_peft_model import HFPeftModelGenerator
from fed_rag.generators.hf_pretrained_model import HFPretrainedModelGenerator


class ModelVariants(str, Enum):
    PLAIN = "plain"
    Q4BIT = "q4bit"
    LORA = "lora"
    QLORA = "qlora"


class ModelRegistry(BaseModel):
    _plain: HFPretrainedModelGenerator | None = PrivateAttr(default=None)
    _q4bit: HFPretrainedModelGenerator | None = PrivateAttr(default=None)
    _lora: HFPeftModelGenerator | None = PrivateAttr(default=None)
    _qlora: HFPeftModelGenerator | None = PrivateAttr(default=None)

    def __init__(
        self,
        plain: HFPretrainedModelGenerator | None = None,
        q4bit: HFPretrainedModelGenerator | None = None,
        lora: HFPeftModelGenerator | None = None,
        qlora: HFPeftModelGenerator | None = None,
    ):
        super().__init__()
        self._plain = plain
        self._q4bit = q4bit
        self._lora = lora
        self._qlora = qlora

    def __getitem__(
        self, key: str
    ) -> HFPeftModelGenerator | HFPretrainedModelGenerator:
        match key:
            case ModelVariants.PLAIN:
                retval = self._plain
            case ModelVariants.Q4BIT:
                retval = self._q4bit
            case ModelVariants.LORA:
                retval = self._lora
            case ModelVariants.QLORA:
                retval = self._qlora
            case _:
                raise ValueError(f"Invalid variant {key}.")
        if retval is None:
            raise ValueError(f"Variant {key} has not been specified")
        return retval
