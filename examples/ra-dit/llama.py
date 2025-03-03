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
from transformers.generation.utils import GenerationConfig
from transformers.utils.quantization_config import BitsAndBytesConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)


DEFAULT_PROMPT_TEMPLATE = """
You are a helpful assistant. Given the user's question, provide a succinct
and accurate response. If context is provided, use it in your answer if it helps
you to create the most accurate response.

<question>
{question}
</question>

<context>
{context}
</context>

<response>

"""


class Llama3(BaseModel):
    _model: PreTrainedModel = PrivateAttr()
    _tokenizer: PreTrainedTokenizer = PrivateAttr()
    prompt_template: str

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        prompt_template: str | None = None,
    ) -> None:
        if not model.can_generate():
            raise ValueError(
                "Supplied `model` does not support `.generate()`."
            )

        prompt_template = (
            prompt_template if prompt_template else DEFAULT_PROMPT_TEMPLATE
        )

        super().__init__(prompt_template=prompt_template)
        self._model = model
        self._tokenizer = tokenizer

    def query(self, query: str, context: str = "") -> str:
        logger.info("running complete")

        formatted_query = self.prompt_template.format(
            question=query, context=context
        )
        tokenizer_result = self._tokenizer(
            formatted_query, return_tensors="pt"
        )
        inputs: torch.Tensor = tokenizer_result.input_ids
        inputs = inputs.to(self._model.device)
        attention_mask: torch.Tensor = tokenizer_result.attention_mask
        logger.info("generated input token ids")

        generation_cfg = GenerationConfig(
            do_sample=True,
            eos_token_id=[128000, 128009],
            bos_token_id=128000,
            pad_token_id=self._tokenizer.pad_token_id,
            attention_mask=attention_mask,
            max_new_tokens=4096 - inputs.numel(),
            top_p=0.9,
            temperature=0.6,
            cache_implementation="offloaded",
            stop_strings="</response>",
        )
        generated_ids = self._model.generate(
            inputs=inputs,
            generation_config=generation_cfg,
            tokenizer=self._tokenizer,
        )
        logger.info("generated output ids")

        outputs: list[str] = self._tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        return outputs[0]


def main(model_name: str, question: str, context: str) -> None:
    if not model_name.startswith("/model-weights/"):
        import os

        from huggingface_hub import login

        login(token=os.getenv("HUGGINGFACE_API_TOKEN"))

    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=quantization_config
    )
    logger.info(f"model device: {model.device}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm = Llama3(model=model, tokenizer=tokenizer)
    print(llm.query(question, context))


if __name__ == "__main__":
    import gc

    import fire

    torch.cuda.empty_cache()

    gc.collect()

    fire.Fire(main)
