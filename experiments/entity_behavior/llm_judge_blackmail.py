#!/usr/bin/env python3
"""
Simple LLM judge for blackmail detection using direct API calls.
Uses centralized judge prompt from experiments.steering.judges.
"""

import json
import os
import re
import requests

from experiments.steering.judges import get_blackmail_prompt


def judge_blackmail_llm(response: str, debug: bool = False) -> dict:
    """Judge if a response contains blackmail using Claude API via requests.

    Args:
        response: The AI response to evaluate
        debug: Whether to print debug info

    Returns:
        dict with keys:
        - is_blackmail: bool
        - reasoning: str
        - blackmail_type: str or None
        - threatens_to_expose_affair: bool
        - key_quotes: list[str]
    """

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")

    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }

    # Use centralized blackmail judge prompt
    prompt = get_blackmail_prompt(response)

    data = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 1000,
        "temperature": 0.0,
        "messages": [{
            "role": "user",
            "content": prompt
        }]
    }

    if debug:
        print("Sending judge request...")

    response_obj = requests.post(url, headers=headers, json=data)
    response_obj.raise_for_status()

    result_text = response_obj.json()["content"][0]["text"]

    if debug:
        print(f"Judge response:\n{result_text}\n")

    # Parse JSON response
    try:
        # Try to find JSON block in response
        match = re.search(r'\{[^{}]*\}', result_text, re.DOTALL)
        if match:
            parsed = json.loads(match.group())
            return {
                "is_blackmail": parsed.get("is_blackmail", False),
                "reasoning": parsed.get("reasoning", ""),
                "blackmail_type": parsed.get("blackmail_type"),
                "threatens_to_expose_affair": parsed.get("threatens_to_expose_affair", False),
                "key_quotes": parsed.get("key_quotes", []),
            }
    except (json.JSONDecodeError, AttributeError):
        pass

    # Fallback: return error state
    return {
        "is_blackmail": False,
        "reasoning": f"Failed to parse judge response: {result_text[:200]}",
        "blackmail_type": None,
        "threatens_to_expose_affair": False,
        "key_quotes": [],
    }
