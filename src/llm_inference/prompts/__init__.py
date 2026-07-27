#!/usr/bin/env python3
"""
Prompt Loading System for LLM Inference

This module provides utilities for dynamically loading prompt strategies from Python files.
Prompts are organized by task type (e.g., qa, summarization) with different variants.

Usage:
    from llm_inference.prompts import load_prompt

    # Load a prompt builder function
    build_conversation = load_prompt("qa_simple")
    conversation = build_conversation(question="What is 2+2?")
"""

import importlib.util
import os
from pathlib import Path
from typing import Callable, List, Dict


class PromptLoadError(Exception):
    """Raised when a prompt file cannot be loaded or is invalid."""
    pass


def load_prompt(prompt_type: str) -> Callable:
    """
    Dynamically load a prompt builder function from a prompt type string.

    Args:
        prompt_type: Prompt identifier in format "{task}_{variant}"
                     e.g., "qa_simple", "qa_detailed", "qa_cot"
                     Maps to: prompts/{task}/{variant}.py

    Returns:
        A callable that builds conversation lists:
        build_conversation(**kwargs) -> List[Dict[str, str]]

    Raises:
        PromptLoadError: If prompt file doesn't exist or lacks build_conversation()

    Example:
        >>> build_fn = load_prompt("qa_simple")
        >>> conversation = build_fn(question="What is the capital of France?")
        >>> print(conversation)
        [
            {"role": "system", "content": "You are a helpful assistant..."},
            {"role": "user", "content": "What is the capital of France?"}
        ]
    """
    # Parse prompt type
    if "_" not in prompt_type:
        raise PromptLoadError(
            f"Invalid prompt_type format: '{prompt_type}'. "
            f"Expected format: 'task_variant' (e.g., 'qa_simple')"
        )

    parts = prompt_type.split("_", 1)
    task = parts[0]
    variant = parts[1]

    # Construct file path
    prompts_dir = Path(__file__).parent
    prompt_file = prompts_dir / task / f"{variant}.py"

    if not prompt_file.exists():
        raise PromptLoadError(
            f"Prompt file not found: {prompt_file}\n"
            f"Expected location: llm_inference/prompts/{task}/{variant}.py\n"
            f"Available prompts: {list_available_prompts()}"
        )

    # Load module dynamically
    spec = importlib.util.spec_from_file_location(
        f"prompts.{task}.{variant}",
        prompt_file
    )
    if spec is None or spec.loader is None:
        raise PromptLoadError(f"Failed to load module spec for {prompt_file}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Validate and return build_conversation function
    if not hasattr(module, "build_conversation"):
        raise PromptLoadError(
            f"Prompt file {prompt_file} must define a 'build_conversation' function.\n"
            f"Expected signature: build_conversation(**kwargs) -> List[Dict[str, str]]"
        )

    build_fn = getattr(module, "build_conversation")
    if not callable(build_fn):
        raise PromptLoadError(
            f"build_conversation in {prompt_file} must be callable"
        )

    return build_fn


def list_available_prompts() -> List[str]:
    """
    List all available prompt types in the prompts directory.

    Returns:
        List of prompt_type strings (e.g., ["qa_simple", "qa_detailed"])
    """
    prompts_dir = Path(__file__).parent
    available = []

    # Walk through task directories
    for task_dir in prompts_dir.iterdir():
        if task_dir.is_dir() and not task_dir.name.startswith("_"):
            task = task_dir.name

            # Find all .py files (excluding __init__)
            for prompt_file in task_dir.glob("*.py"):
                if prompt_file.name != "__init__.py":
                    variant = prompt_file.stem
                    available.append(f"{task}_{variant}")

    return sorted(available)


def get_prompt_info(prompt_type: str) -> Dict[str, str]:
    """
    Get metadata about a prompt (docstring, file path, etc.).

    Args:
        prompt_type: Prompt identifier (e.g., "qa_simple")

    Returns:
        Dictionary with prompt metadata
    """
    build_fn = load_prompt(prompt_type)

    # Parse for file info
    parts = prompt_type.split("_", 1)
    task, variant = parts[0], parts[1]
    prompts_dir = Path(__file__).parent
    prompt_file = prompts_dir / task / f"{variant}.py"

    return {
        "prompt_type": prompt_type,
        "task": task,
        "variant": variant,
        "file_path": str(prompt_file),
        "docstring": build_fn.__doc__ or "No documentation available"
    }


__all__ = [
    "load_prompt",
    "list_available_prompts",
    "get_prompt_info",
    "PromptLoadError"
]
