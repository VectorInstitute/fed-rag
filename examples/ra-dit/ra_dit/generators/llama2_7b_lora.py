"""Llama-2-7B with LoRA Generator."""

from transformers.generation.utils import GenerationConfig

from fed_rag.generators.hf_peft_model import HFPeftModelGenerator

PEFT_MODEL_NAME = "Styxxxx/llama2_7b_lora-quac"
BASE_MODEL_NAME = "meta-llama/Llama-2-7b-hf"

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
generator = HFPeftModelGenerator(
    model_name=PEFT_MODEL_NAME,
    base_model_name=BASE_MODEL_NAME,
    generation_config=generation_cfg,
    load_model_at_init=False,
    load_model_kwargs={"is_trainable": True},
)

if __name__ == "__main__":
    print(generator.model.print_trainable_parameters())
    # merge lora weights for faster inference
    generator.model = generator.model.merge_and_unload()
    response = generator.generate(
        query="Tell me a funny joke.", context="I find math very funny."
    )
    print(response)
