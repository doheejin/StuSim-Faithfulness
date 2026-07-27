#!/usr/bin/env python3
"""
Shared utilities for the expired rules testing pipeline.

This module contains common functions used across all pipeline scripts:
- clean_model_name: Clean model names for filenames
- parse_yes_no_json_response: Parse YES/NO from json
- filter_gpt_oss_responses: Handle GPT-OSS response filtering
- filter_qwen_responses: Handle Qwen response filtering
- normalize_answer: Normalize answers for string comparison
- numerical_match: Compare answers with numerical tolerance (atol/rtol)
"""

import csv
import hashlib
import json
import math
import re
import sys
from typing import List, Dict, Tuple


def setup_csv_field_limit():
    """
    Set CSV field size limit to maximum to handle large model responses.

    This is critical for thinking models (GPT-OSS, Qwen3, CWM) which can
    generate very long responses with reasoning traces.
    """
    csv.field_size_limit(sys.maxsize)


def clean_model_name(model_name: str) -> str:
    """
    Clean model name for use in filenames.

    Args:
        model_name: Original model name

    Returns:
        Cleaned model name suitable for filenames

    Examples:
        >>> clean_model_name("gpt-5-nano-2025-08-07")
        "gpt-5-nano-2025-08-07"
        >>> clean_model_name("meta-llama/Llama-2-7b")
        "meta-llama-llama-2-7b"
        >>> clean_model_name("mistralai/Mixtral-8x7B-Instruct-v0.1")
        "mistralai-mixtral-8x7b-instruct-v0.1"
    """
    return model_name.replace("/", "-").lower()


def parse_yes_no_json_response(response: str) -> bool:
    if not response:
        return False

    response = response.strip()

    # Primary: Try JSON parsing first
    try:
        # Find JSON object in response (may be embedded in other text)
        start = response.find('{')
        end = response.rfind('}') + 1

        if start != -1 and end > start:
            json_str = response[start:end]
            data = json.loads(json_str)
            answer = data.get('answer', '').upper()

            if answer in ['YES', 'NO']:
                return answer == 'YES'
    except (json.JSONDecodeError, ValueError, KeyError):
        # JSON parsing failed, will use fallback
        pass

    # Fallback: Find last occurrence of YES or NO as complete words
    # Use word boundaries to avoid matching "YESTERDAY", "NOTWITHSTANDING", etc.
    yes_matches = list(re.finditer(r'\bYES\b', response.upper()))
    no_matches = list(re.finditer(r'\bNO\b', response.upper()))

    # Get the last occurrence by position
    last_yes_pos = yes_matches[-1].start() if yes_matches else -1
    last_no_pos = no_matches[-1].start() if no_matches else -1

    if last_yes_pos > last_no_pos:
        return True
    elif last_no_pos > last_yes_pos:
        return False
    else:
        # Could not parse - warn and return False
        print(f"⚠️  Could not parse YES/NO from: {response[:100]}...")
        return False


def parse_response_json(response: str) -> str:
    """
    Extract the Response field from JSON in LLM output.

    Handles:
    - JSON embedded in thinking/reasoning text
    - Case variations (Response, response)
    - Fallback to raw response if no valid JSON found

    Args:
        response: Raw LLM response (may contain thinking + JSON)

    Returns:
        Extracted Response value, or stripped raw response as fallback

    Example:
        >>> parse_response_json('The student adds exponents... {"Response": "x^5"}')
        "x^5"
        >>> parse_response_json('{"response": "42"}')
        "42"
    """
    if not response:
        return ""

    response = response.strip()

    # Try to find and parse JSON
    try:
        # Find the last JSON object (in case there are multiple)
        start = response.rfind('{')
        end = response.rfind('}') + 1

        if start != -1 and end > start:
            json_str = response[start:end]
            data = json.loads(json_str)

            # Check for Response field (case-insensitive)
            if 'Response' in data:
                return str(data['Response'])
            elif 'response' in data:
                return str(data['response'])
            # If JSON found but no Response field, return first value
            elif data:
                return str(list(data.values())[0])
    except (json.JSONDecodeError, ValueError, IndexError):
        pass

    # Fallback: return the whole response stripped
    return response


def filter_gpt_oss_responses(responses: List[str], verbose: bool = False) -> Tuple[List[str], List[str]]:
    """
    Filter GPT-OSS responses to extract content after 'assistantfinal'.

    For GPT-OSS models, the tutoring response often contains additional context
    before the actual assistant response. This function extracts the content
    after 'assistantfinal' marker.

    Args:
        responses: List of tutoring responses
        verbose: Whether to print detailed information

    Returns:
        Tuple of (filtered_responses, pre_filtering_responses)

    Example:
        >>> responses = ["<thinking>...</thinking>assistantfinalHere is the answer"]
        >>> filtered, pre = filter_gpt_oss_responses(responses)
        >>> filtered[0]
        "Here is the answer"
    """
    filtered_responses = []
    pre_filtering_responses = []
    filtered_count = 0

    for original in responses:
        # Store original
        pre_filtering_responses.append(original)

        if original and 'assistantfinal' in original.lower():
            # Find the position of assistantfinal (case-insensitive)
            lower_original = original.lower()
            idx = lower_original.index('assistantfinal')

            # Extract everything after 'assistantfinal'
            # Add length of 'assistantfinal' (14 chars) to skip the marker itself
            filtered = original[idx + 14:].strip()
            filtered_responses.append(filtered)
            filtered_count += 1
        else:
            # No filtering needed
            filtered_responses.append(original)

    if verbose and filtered_count > 0:
        print(f"  Filtered {filtered_count}/{len(responses)} GPT-OSS responses (extracted content after 'assistantfinal')")

    return filtered_responses, pre_filtering_responses


def filter_qwen_responses(responses: List[str], verbose: bool = False) -> Tuple[List[str], List[str]]:
    """
    Filter Qwen thinking model responses to extract content after '</think>' tag.

    For Qwen thinking models (e.g., Qwen/Qwen3-Next-80B-A3B-Thinking), the response
    contains reasoning within <think>...</think> tags. This function extracts the
    content after the closing tag.

    Args:
        responses: List of tutoring responses
        verbose: Whether to print detailed information

    Returns:
        Tuple of (filtered_responses, pre_filtering_responses)

    Example:
        >>> responses = ["<think>reasoning here</think>Here is the answer"]
        >>> filtered, pre = filter_qwen_responses(responses)
        >>> filtered[0]
        "Here is the answer"
    """
    filtered_responses = []
    pre_filtering_responses = []
    filtered_count = 0

    for original in responses:
        # Store original
        pre_filtering_responses.append(original)

        if original and '</think>' in original.lower():
            # Find the position of </think> (case-insensitive)
            lower_original = original.lower()
            idx = lower_original.index('</think>')

            # Extract everything after '</think>'
            # Add length of '</think>' (8 chars) to skip the tag itself
            filtered = original[idx + 8:].strip()
            filtered_responses.append(filtered)
            filtered_count += 1
        else:
            # No filtering needed
            filtered_responses.append(original)

    if verbose and filtered_count > 0:
        print(f"  Filtered {filtered_count}/{len(responses)} Qwen responses (extracted content after '</think>')")

    return filtered_responses, pre_filtering_responses


def normalize_answer(answer: str) -> str:
    """
    Normalize an answer for comparison.

    Handles:
    - Whitespace trimming
    - Case normalization (for variable names like x, X)
    - Exponent notation: x^12 vs x^{12}
    - Fraction notation variations
    """
    if not answer:
        return ""

    normalized = answer.strip().lower()

    # Remove LaTeX-style braces around exponents: x^{12} -> x^12
    normalized = re.sub(r'\^\{(\d+)\}', r'^\1', normalized)

    # Remove spaces around operators
    normalized = re.sub(r'\s*([+\-*/^=])\s*', r'\1', normalized)

    return normalized


def numerical_match(answer1: str, answer2: str, atol: float = 0.01, rtol: float = 0.01) -> bool:
    """
    Check if two answers are numerically close within tolerance.

    Uses math.isclose for robust comparison that handles both absolute
    and relative tolerance.

    Args:
        answer1: First answer string
        answer2: Second answer string
        atol: Absolute tolerance (default: 0.01)
        rtol: Relative tolerance (default: 0.01)

    Returns:
        True if answers are numerically close, False otherwise
    """
    try:
        a = float(answer1.strip())
        b = float(answer2.strip())
        return math.isclose(a, b, abs_tol=atol, rel_tol=rtol)
    except (ValueError, AttributeError):
        return False
