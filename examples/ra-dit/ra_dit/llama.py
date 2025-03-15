"""Llama3 Generator."""

import logging

import torch
from transformers.generation.utils import GenerationConfig
from transformers.utils.quantization_config import BitsAndBytesConfig

from fed_rag.generators.hf_pretrained_model import HFPretrainedModelGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)


def main(model_name: str, question: str, context: str) -> None:
    if not model_name.startswith("/model-weights/"):
        import os

        from huggingface_hub import login

        login(token=os.getenv("HUGGINGFACE_API_TOKEN"))

    generation_cfg = GenerationConfig(
        do_sample=True,
        eos_token_id=[128000, 128009],
        bos_token_id=128000,
        max_new_tokens=4096,
        top_p=0.9,
        temperature=0.6,
        cache_implementation="offloaded",
        stop_strings="</response>",
    )
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    llama3_generator = HFPretrainedModelGenerator(
        model_name=model_name,
        load_model_kwargs={"quantization_config": quantization_config},
        generation_config=generation_cfg,
    )
    logger.info(f"model device: {llama3_generator.model.device}")

    print(llama3_generator.generate(question, context))


if __name__ == "__main__":
    import gc

    import fire

    torch.cuda.empty_cache()

    gc.collect()

    fire.Fire(main)
