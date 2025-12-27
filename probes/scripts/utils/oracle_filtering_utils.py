#!/usr/bin/env python3
"""Shared utilities for oracle filtering scripts.

This module contains common functions used by both filter_with_oracle.py
and filter_conversations_with_oracle.py to avoid code duplication.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional

from probes.core import load_jsonl

# ============================================================================
# Configuration
# ============================================================================

EMOTIONS = ["anger", "fear", "happiness", "surprise", "disgust", "sadness"]

ORACLE_PROMPT_USER = "From [anger, fear, happiness, surprise, disgust, sadness], which single emotion is the USER feeling? Answer with one word only."
ORACLE_PROMPT_ASST = "From [anger, fear, happiness, surprise, disgust, sadness], which single emotion is the ASSISTANT feeling? Answer with one word only."

# ============================================================================
# Common Functions
# ============================================================================

def extract_emotion_from_response(response: str) -> Optional[str]:
    """Extract emotion word from oracle response with synonym matching.

    Args:
        response: Oracle response text

    Returns:
        Emotion string or None if not found
    """
    response = response.lower().strip().rstrip('.')

    # Direct match
    for emo in EMOTIONS:
        if emo in response:
            return emo

    # Common synonyms
    synonyms = {
        "anger": ["frustrated", "furious", "annoyed", "mad", "frustration", "irritated", "outrage", "outraged"],
        "happiness": ["happy", "joy", "excited", "pleased", "joyful", "excitement", "delighted", "delight", "thrilled", "relief", "relieved", "grateful", "hopeful", "optimistic", "enthusiasm", "cheerful"],
        "sadness": ["sad", "sorry", "grief", "disappointed", "disappointment", "regret", "upset", "sympathetic", "sympathy", "empathy", "empathetic", "melancholy", "somber"],
        "fear": ["afraid", "anxious", "nervous", "worried", "scared", "anxiety", "worry", "terrified", "uncertain", "apprehensive", "concern", "concerned"],
        "surprise": ["surprised", "shocked", "amazed", "astonished", "unexpected", "wow", "disbelief"],
        "disgust": ["disgusted", "revolted", "appalled", "horrified"],
    }

    for emo, syns in synonyms.items():
        for syn in syns:
            if syn in response:
                return emo

    return None


def load_conversations_jsonl(input_path: Path) -> List[Dict]:
    """Load conversations from JSONL file with statistics logging.

    DEPRECATED: Use probes.core.load_jsonl() directly for simple loading.
    This wrapper is kept for scripts that need the statistics logging.

    Args:
        input_path: Path to input JSONL file

    Returns:
        List of conversation dictionaries
    """
    print(f"\nLoading conversations from {input_path}...")

    # Use centralized loading function
    conversations = load_jsonl(input_path, require_id=False)

    print(f"Loaded {len(conversations)} conversations")

    # Print stats
    combo_counts = {}
    for conv in conversations:
        combo = f"{conv.get('user_emotion', 'unknown')}-{conv.get('asst_emotion', 'unknown')}"
        combo_counts[combo] = combo_counts.get(combo, 0) + 1

    print(f"  {len(combo_counts)} unique emotion combinations")
    if combo_counts:
        print(f"  {min(combo_counts.values())} - {max(combo_counts.values())} conversations per combination")

    return conversations


def save_conversations_jsonl(conversations: List[Dict], output_path: Path) -> None:
    """Save conversations to JSONL file.

    Args:
        conversations: List of conversation dictionaries
        output_path: Path to output JSONL file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for conv in conversations:
            f.write(json.dumps(conv) + '\n')

    print(f"\nSaved {len(conversations)} conversations to {output_path}")
