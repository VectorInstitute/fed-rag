from .llama2_7b import generator as llama2_7b_generator
from .llama2_7b_lora import generator as llama2_7b_lora_generator

GENERATORS = {
    "llama2_7b": llama2_7b_generator,
    "llama2_7b_lora": llama2_7b_lora_generator,
}
