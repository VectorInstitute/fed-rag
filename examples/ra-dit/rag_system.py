"""RA-DIT original RAG System."""

from transformers.generation.utils import GenerationConfig
from transformers.utils.quantization_config import BitsAndBytesConfig

from fed_rag.generators.hf_pretrained_model import HFPretrainedModelGenerator
from fed_rag.retrievers.hf_sentence_transformer import (
    HFSentenceTransformerRetriever,
)

# Build a rag system

## retriever
dragon_retriever = HFSentenceTransformerRetriever(
    query_model_name="nthakur/dragon-plus-query-encoder",
    context_model_name="nthakur/dragon-plus-context-encoder",
)

## generator
model_name = ...
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
