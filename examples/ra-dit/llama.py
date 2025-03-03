"""Llama LLM."""

import logging

import torch
from pydantic import BaseModel, PrivateAttr
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)


class LLM(BaseModel):
    _model: PreTrainedModel = PrivateAttr()
    _tokenizer: PreTrainedTokenizer = PrivateAttr()

    def __init__(
        self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer
    ) -> None:
        if not model.can_generate():
            raise ValueError(
                "Supplied `model` does not support `.generate()`."
            )

        super().__init__()
        self._model = model
        self._tokenizer = tokenizer

    def complete(self, query: str) -> str:
        logger.info("running complete")
        inputs: torch.Tensor = self._tokenizer(
            query, return_tensors="pt"
        ).input_ids
        inputs = inputs.to(self._model.device)
        logger.info("generated input token ids")
        generated_ids = self._model.generate(inputs=inputs, max_new_tokens=500)
        logger.info("generated output ids")

        outputs: list[str] = self._tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        return outputs[0]


def main(model_name: str, prompt: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to(device)
    logger.info(f"model device: {model.device}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm = LLM(model=model, tokenizer=tokenizer)
    print(llm.complete(prompt))


if __name__ == "__main__":
    import os

    import fire
    from huggingface_hub import login

    login(token=os.getenv("HUGGINGFACE_API_TOKEN"))

    fire.Fire(main)
