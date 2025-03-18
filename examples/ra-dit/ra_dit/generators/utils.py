"""Utils module."""

from typing import TypedDict

from fed_rag.generators.hf_peft_model import HFPeftModelGenerator
from fed_rag.generators.hf_pretrained_model import HFPretrainedModelGenerator


class ModelVariants(TypedDict):
    plain: HFPretrainedModelGenerator
    lora: HFPeftModelGenerator
    qlora: HFPeftModelGenerator
