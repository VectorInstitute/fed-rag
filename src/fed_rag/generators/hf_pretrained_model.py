"""HuggingFace PretrainedModel Generator"""

from typing import Any

from pydantic import Field

from fed_rag.base.generator import BaseGenerator


class HFPretrainedModelGenerator(BaseGenerator):
    model_name: str = Field(
        description="Name of HuggingFace model. Used for loading the model from HF hub or local."
    )
    generation_config: Any = Field(
        description="The generation config used for generating with the PreTrainedModel."
    )
