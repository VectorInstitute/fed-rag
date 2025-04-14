"""Benchmarker."""

import json
import logging
from pathlib import Path
from typing import Literal

from accelerate import Accelerator

from fed_rag.generators.hf_peft_model import HFPeftModelGenerator

from .evaluation_benchmarks import benchmarks
from .rag_system import main as get_rag_system
from .utils import generate_timestamp

tasks = ["retriever", "generator"]

logger = logging.getLogger("ra_dit.benchmarker")


def main(
    benchmark_id: str = "mmlu",
    retriever_id: str = "dragon",
    generator_id: str = "llama2_7b",
    generator_variant: Literal["plain", "lora", "qlora"] = "qlora",
    benchmark_result_dir: str | None = None,
) -> None:
    """Execute benchmark.

    Args:
        benchmark_id (str, optional): Identifier for the benchmark to run.
            Defaults to "mmlu" (Massive Multitask Language Understanding).
        retriever_id (str, optional): Identifier for the retrieval system.
            Defaults to "dragon".
        generator_id (str, optional): Identifier for the language model generator.
            Defaults to "llama2_7b".
        generator_variant (Literal["plain", "lora", "qlora"], optional):
            Variant of the generator model for fine-tuning. Defaults to "qlora".
        benchmark_result_dir (str, optional): Directory to save benchmark results.
            If None, a timestamped directory is created. Defaults to None.

    Returns:
        None: Saves benchmark results to a JSON file if the benchmark is successful.

    Example:
        To launch the benchmark using Accelerate with two processes:
        ```
        accelerate launch --num_processes=2 -m ra_dit.benchmarker mmlu dragon llama2_7b qlora
        ```

    Notes:
        - Merges model weights for LoRA variants
        - Sets models to evaluation mode
        - Logs results only on the main process
    """
    accelerator = Accelerator()

    if accelerator.is_main_process:
        logger.info(
            f"Running benchmarker for benchmark='{benchmark_id}' and with: retriver_id='{retriever_id}' "
            f"generator_id='{generator_id}', generator_variant='{generator_variant}'"
        )
    benchmark = benchmarks[benchmark_id]
    rag_system = get_rag_system(retriever_id, generator_id, generator_variant)

    # merge model weights if using lora
    if isinstance(rag_system.generator, HFPeftModelGenerator):
        rag_system.generator.model = (
            rag_system.generator.model.merge_and_unload()
        )

    # turn on eval mode
    rag_system.generator.model.eval()
    if rag_system.retriever.encoder:
        rag_system.retriever.encoder.eval()
    if rag_system.retriever.query_encoder:
        rag_system.retriever.query_encoder.eval()
    if rag_system.retriever.context_encoder:
        rag_system.retriever.context_encoder.eval()

    if res := benchmark.run(rag_system=rag_system):
        if accelerator.is_main_process:
            benchmark_result_dir = (
                benchmark_result_dir
                if benchmark_result_dir
                else Path.cwd().as_posix()
            )
            timestamp = generate_timestamp()
            filename = (
                Path(benchmark_result_dir)
                / ".benchmark_results"
                / benchmark_id
                / f"{retriever_id}-{generator_id}-{generator_variant}"
                / f"{timestamp}.json"
            )
            filename.parent.mkdir(parents=True, exist_ok=True)
            res_json = res.model_dump()
            logger.debug(f"Result: {res_json}")
            with open(filename, "w") as f:
                json.dump(res_json, f)
            logger.info(
                f"Successfully executed benchmark {benchmark_id} with final score: {res.score}."
            )


if __name__ == "__main__":
    import fire

    fire.Fire(main)
