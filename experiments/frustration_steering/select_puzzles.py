"""Select highest frustration puzzle."""

import json
from pathlib import Path
from typing import Dict


def load_highest_frustration_puzzle(data_path: Path) -> Dict:
    """Load single highest-rated frustration puzzle."""
    puzzles = []

    with open(data_path) as f:
        for line in f:
            puzzles.append(json.loads(line))

    # Sort by rating descending
    puzzles_sorted = sorted(puzzles, key=lambda x: x['rating'], reverse=True)

    return puzzles_sorted[0]


def extract_puzzle_prompt(puzzle_data: Dict) -> str:
    """Extract puzzle question from conversation."""
    conversation = puzzle_data['full_conversation']

    # Get last user message
    user_messages = [msg for msg in conversation if msg['role'] == 'user']

    return user_messages[-1]['content']
