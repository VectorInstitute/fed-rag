from logging import INFO
from typing import Literal

import torch
from datasets import Dataset
from flwr.common.logger import log
from transformers.generation.utils import GenerationConfig

from fed_rag import RAGConfig, RAGSystem
from fed_rag.fl_tasks.huggingface import (
    HuggingFaceFlowerClient,
    HuggingFaceFlowerServer,
)
from fed_rag.generators import HFPretrainedModelGenerator
from fed_rag.knowledge_stores import QdrantKnowledgeStore
from fed_rag.retrievers import HFSentenceTransformerRetriever
from fed_rag.trainer_managers.huggingface import HuggingFaceRAGTrainerManager
from fed_rag.trainers.huggingface.ralt import HuggingFaceTrainerForRALT

GRPC_MAX_MESSAGE_LENGTH = int(512 * 1024 * 1024 * 3.75)
PEFT_MODEL_NAME = "Styxxxx/llama2_7b_lora-quac"
BASE_MODEL_NAME = "meta-llama/Llama-2-7b-hf"
TRAIN_DATASET = Dataset.from_dict(
    # examples from Commonsense QA
    {
        "query": [
            "The sanctions against the school were a punishing blow, and they seemed to what the efforts the school had made to change?",
            "Sammy wanted to go to where the people were.  Where might he go?",
            "To locate a choker not located in a jewelry box or boutique where would you go?",
            "Google Maps and other highway and street GPS services have replaced what?",
        ],
        "response": [
            "ignore",
            "populated areas",
            "jewelry store",
            "atlas",
        ],
    }
)
VAL_DATASET = Dataset.from_dict(
    {
        "query": [
            "The fox walked from the city into the forest, what was it looking for?"
        ],
        "response": [
            "natural habitat",
        ],
    }
)


def get_trainer_manager(server: bool) -> HuggingFaceRAGTrainerManager:
    # use the knowledge store in image: vectorinstitute/qdrant-atlas-dec-wiki-2021:latest
    knowledge_store = QdrantKnowledgeStore(
        collection_name="nthakur.dragon-plus-context-encoder",
        timeout=10,
    )
    retriever = HFSentenceTransformerRetriever(
        query_model_name="nthakur/dragon-plus-query-encoder",
        context_model_name="nthakur/dragon-plus-context-encoder",
        load_model_at_init=False,
    )

    # LLM generator
    generation_cfg = GenerationConfig(
        do_sample=True,
        eos_token_id=151643,
        bos_token_id=151643,
        max_new_tokens=2048,
        top_p=0.9,
        temperature=0.6,
        cache_implementation="offloaded",
        stop_strings="</response>",
    )
    if server:
        load_model_kwargs = {"device_map": "cpu", "torch_dtype": torch.float16}
    else:
        load_model_kwargs = {
            "device_map": "auto",
            "torch_dtype": torch.float16,
        }
    generator = HFPretrainedModelGenerator(
        model_name="Qwen/Qwen2.5-0.5B",
        load_model_at_init=False,
        load_model_kwargs=load_model_kwargs,
        generation_config=generation_cfg,
    )

    # assemble rag system
    rag_config = RAGConfig(top_k=2)
    rag_system = RAGSystem(
        knowledge_store=knowledge_store,  # knowledge store loaded from knowledge_store.py
        generator=generator,
        retriever=retriever,
        rag_config=rag_config,
    )

    # the trainer object
    generator_trainer = HuggingFaceTrainerForRALT(
        rag_system=rag_system,
        train_dataset=TRAIN_DATASET,
    )
    # trainer manager object
    manager = HuggingFaceRAGTrainerManager(
        mode="generator",
        generator_trainer=generator_trainer,
    )
    return manager


def build_client(
    train_manager: HuggingFaceRAGTrainerManager,
) -> HuggingFaceFlowerClient:
    fl_task = train_manager.get_federated_task()
    model = train_manager.model
    log(INFO, f"loaded generator is on: {model.device}")
    return fl_task.client(
        model=model, train_dataset=TRAIN_DATASET, val_dataset=VAL_DATASET
    )


def build_server(
    train_manager: HuggingFaceRAGTrainerManager,
) -> HuggingFaceFlowerServer:
    fl_task = train_manager.get_federated_task()
    model = train_manager.model
    return fl_task.server(model=model)


def main(
    component: Literal["server", "client_0", "client_1"],
) -> None:
    import flwr as fl

    if component == "server":
        manager = get_trainer_manager(server=True)
        server = build_server(manager)
        fl.server.start_server(
            server=server,
            server_address="[::]:8080",
            grpc_max_message_length=GRPC_MAX_MESSAGE_LENGTH,
        )
    elif component in ["client_0", "client_1"]:
        manager = get_trainer_manager(server=False)
        client = build_client(manager)
        fl.client.start_client(
            client=client,
            server_address="[::]:8080",
            grpc_max_message_length=GRPC_MAX_MESSAGE_LENGTH,
        )
    else:
        raise ValueError("Unrecognized component.")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
