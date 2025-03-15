"""Llama-2-7B Generator."""

from transformers.generation.utils import GenerationConfig

from fed_rag.generators.hf_pretrained_model import HFPretrainedModelGenerator

MODEL_NAME = "meta-llama/Llama-2-7b-hf"

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
generator = HFPretrainedModelGenerator(
    model_name=MODEL_NAME,
    generation_config=generation_cfg,
    load_model_at_init=False,
)

if __name__ == "__main__":
    response = generator.generate(
        query="Tell me a funny joke.", context="I find math very funny."
    )
    print(response)
