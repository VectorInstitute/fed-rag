"""Unsloth FastModel Generator"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    pass


# since Unsloth is a patch on HuggingFace models and tokenizers


DEFAULT_PROMPT_TEMPLATE = """
You are a helpful assistant. Given the user's query, provide a succinct
and accurate response. If context is provided, use it in your answer if it helps
you to create the most accurate response.

<query>
{query}
</query>

<context>
{context}
</context>

<response>

"""
