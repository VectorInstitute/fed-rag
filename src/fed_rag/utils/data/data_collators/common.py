"""Common utility functions for data collators."""

import torch
import torch.nn.functional as F

from fed_rag.base.generator import BaseGenerator


def compute_target_sequence_proba(
    generator: BaseGenerator, prompt: str, target: str
) -> torch.Tensor:
    """Computes the target sequence probability given the prompt.

    Args:
        generator (BaseGenerator): The generator LLM
        prompt (str): The input i.e. conditional prompt sequence
        target (str): The target sequence

    Returns:
        proba (torch.Tensor): The probability of target sequence given a prompt.
            i.e., P_{LLM}(target | sequence)
    """
    # get model
    model = generator.model
    tokenizer = generator.tokenizer

    # Combine prompt and target for teacher forcing
    input_text = prompt + target
    encode_result = tokenizer.encode(input_text)
    input_ids = encode_result["input_ids"]
    print(f"input_ids: {input_ids}")

    # Get the token IDs for just the target portion
    prompt_only_encode_result = tokenizer.encode(prompt)
    prompt_only_input_ids = prompt_only_encode_result["input_ids"]
    print(f"prompt_only_input_ids: {prompt_only_input_ids}")
    target_start_idx = len(prompt_only_encode_result["input_ids"])
    print(f"target_start_idx: {target_start_idx}")
    target_ids = input_ids[target_start_idx:]
    print(f"target_ids: {target_ids}")

    # Get the logits from the model
    with torch.no_grad():
        outputs = model(torch.tensor(input_ids).unsqueeze(0))
        logits = outputs.logits
        print(f"logits: {logits}")

    # Calculate probability of each target token given the previous tokens
    log_probs = []
    for i, target_id in enumerate(target_ids):
        # get log prob of next target token in the sequence
        next_token_pos = target_start_idx + i - 1
        next_token_logits = logits[0, next_token_pos, :]
        probs = F.softmax(next_token_logits, dim=-1)
        log_prob = torch.log(probs[target_id]).item()
        log_probs.append(log_prob)

    # Sum log probabilities to get sequence log probability
    sequence_log_prob = sum(log_probs)
    # Convert to probability
    sequence_prob = torch.exp(torch.tensor(sequence_log_prob)).item()

    return sequence_prob


if __name__ == "__main__":
    from transformers.generation.utils import GenerationConfig

    from fed_rag.generators.hf_pretrained_model import (
        HFPretrainedModelGenerator,
    )

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
        model_name="meta-llama/Llama-2-7b-hf",
        generation_config=generation_cfg,
        load_model_at_init=False,
        load_model_kwargs={"device_map": "auto"},
    )

    prompt = "The capital of France is"
    target = " Toronto"  # Note the space at the beginning

    probability = compute_target_sequence_proba(generator, prompt, target)
    print(f"Probability of '{target}' given '{prompt}': {probability:.6f}")
