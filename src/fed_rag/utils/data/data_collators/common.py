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
    encode_result = tokenizer.encode(input_text, return_tensors="pt")
    input_ids = encode_result["input_ids"]

    # Get the token IDs for just the target portion
    prompt_only_encode_result = tokenizer.encode(
        prompt, add_special_tokens=False
    )
    target_start_idx = len(prompt_only_encode_result["input_ids"])
    target_ids = input_ids[target_start_idx:]

    # Get the logits from the model
    with torch.no_grad():
        logits = model(input_ids)

    # Calculate probability of each target token given the previous tokens
    log_probs = []
    for i, target_id in enumerate(target_ids):
        # get log prob of next target token in the sequence
        next_token_pos = target_start_idx + i - 1
        next_token_logits = logits[next_token_pos, :]
        probs = F.softmax(next_token_logits, dim=-1)
        log_prob = torch.log(probs[target_id]).item()
        log_probs.append(log_prob)

    # Sum log probabilities to get sequence log probability
    sequence_log_prob = sum(log_probs)
    # Convert to probability
    sequence_prob = torch.exp(torch.tensor(sequence_log_prob)).item()

    return sequence_prob
