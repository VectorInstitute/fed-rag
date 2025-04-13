"""Benchmarker."""

# logging
import logging
from typing import Literal
from accelerate import Accelerator

from fed_rag.generators.hf_peft_model import HFPeftModelGenerator

from .evaluation_benchmarks import BenchmarkResult, benchmarks
from .rag_system import main as get_rag_system

tasks = ["retriever", "generator"]

logger = logging.getLogger("ra_dit.benchmarker")


def main(
    benchmark_id: str = "mmlu",
    retriever_id: str = "dragon",
    generator_id: str = "llama2_7b",
    generator_variant: Literal["plain", "lora", "qlora"] = "qlora",
    num_threads: int = 1,
) -> BenchmarkResult:
    accelerator = Accelerator()

    if accelerator.is_main_process:
        logger.info(
            f"Running benchmarker for benchmark='{benchmark_id}' and with: retriver_id='{retriever_id}' "
            f"generator_id='{generator_id}', generator_variant='{generator_variant}' num_threads={num_threads}"
        )
    benchmark = benchmarks[benchmark_id]
    rag_system = get_rag_system(retriever_id, generator_id, generator_variant)

    # merge model weights if using lora
    if isinstance(rag_system.generator, HFPeftModelGenerator):
        rag_system.generator.model = rag_system.generator.model.merge_and_unload()

    # turn on eval mode
    rag_system.generator.model.eval()
    if rag_system.retriever.encoder:
        rag_system.retriever.encoder.eval()
    if rag_system.retriever.query_encoder:
        rag_system.retriever.query_encoder.eval()
    if rag_system.retriever.context_encoder:
        rag_system.retriever.context_encoder.eval()

    res = benchmark.run(rag_system=rag_system, num_threads=num_threads)
    if accelerator.is_main_process:
        logger.info(
            f"Successfully executed benchmark {benchmark_id} with final score: {res.score}."
        )
    return res


if __name__ == "__main__":
    import fire

    fire.Fire(main)
