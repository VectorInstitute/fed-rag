"""Llama-2-7B with LoRA Generator."""

from transformers.generation.utils import GenerationConfig
from transformers.utils.quantization_config import BitsAndBytesConfig

from fed_rag.generators.hf_peft_model import HFPeftModelGenerator
from fed_rag.generators.hf_pretrained_model import HFPretrainedModelGenerator

from .utils import ModelRegistry, ModelVariants

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
quantization_config = BitsAndBytesConfig(load_in_4bit=True)

generator_variants = {
    ModelVariants.PLAIN: HFPretrainedModelGenerator(
        model_name=BASE_MODEL_NAME,
        generation_config=generation_cfg,
        load_model_at_init=False,
        load_model_kwargs={"device_map": "auto"},
    ),
    ModelVariants.Q4BIT: HFPretrainedModelGenerator(
        model_name=BASE_MODEL_NAME,
        generation_config=generation_cfg,
        load_model_at_init=False,
        load_model_kwargs={"device_map": "auto"},
    ),
    ModelVariants.LORA: HFPeftModelGenerator(
        model_name=PEFT_MODEL_NAME,
        base_model_name=BASE_MODEL_NAME,
        generation_config=generation_cfg,
        load_model_at_init=False,
        load_model_kwargs={"is_trainable": True, "device_map": "auto"},
        load_base_model_kwargs={
            "device_map": "auto",
        },
    ),
    ModelVariants.QLORA: HFPeftModelGenerator(
        model_name=PEFT_MODEL_NAME,
        base_model_name=BASE_MODEL_NAME,
        generation_config=generation_cfg,
        load_model_at_init=False,
        load_model_kwargs={"is_trainable": True, "device_map": "auto"},
        load_base_model_kwargs={
            "device_map": "auto",
            "quantization_config": quantization_config,
        },
    ),
}

generator_registry = ModelRegistry(
    **{k.value: v for k, v in generator_variants.items()}
)

if __name__ == "__main__":
    # use qlora
    generator = generator_registry["qlora"]

    # print trainable params
    print(generator.model.print_trainable_parameters())

    # merge lora weights for faster inference
    generator.model = generator.model.merge_and_unload()
    response = generator.generate(
        query="Tell me a funny joke.", context="I find math very funny."
    )
    print(response)
