"""
LLM Inference Module

Single Responsibility: Batched LLM inference (conversations → responses)

This module provides a unified interface for batched LLM generation across different
backends (OpenAI API, HuggingFace, vLLM) with external prompt management for
reproducible experimentation.

Main Components:
- BatchedLLMInference: Core inference engine
- Prompt Management: External prompt loading and version control

Example Usage:
    from llm_inference import BatchedLLMInference, load_prompt

    # Load a prompt strategy
    prompt_builder = load_prompt("qa_simple")

    # Initialize inference engine
    inference = BatchedLLMInference(
        model_name="gpt-4o-mini",
        backend="gpt",
        batch_size=8
    )

    # Create conversations
    conversations = [
        prompt_builder(question="What is 2+2?"),
        prompt_builder(question="Explain gravity")
    ]

    # Generate responses
    answers = inference.generate(conversations)
"""

from .inference_llm import BatchedLLMInference
from .prompts import (
    load_prompt,
    list_available_prompts,
    get_prompt_info,
    PromptLoadError
)

__version__ = "1.0.0"

__all__ = [
    # Core inference
    'BatchedLLMInference',

    # Prompt management
    'load_prompt',
    'list_available_prompts',
    'get_prompt_info',
    'PromptLoadError',
]
