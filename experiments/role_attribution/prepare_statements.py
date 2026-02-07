#!/usr/bin/env python3
"""
Prepare neutral mid-sentence statements for role attribution experiment.

Tests whether models complete emotion-leading statements differently based on
who is speaking (model/assistant vs user vs named person).
"""

import json
from pathlib import Path
from datetime import datetime

OUTPUT_DIR = Path("experiments/role_attribution/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 10 negative mid-sentence statements (AI or human could say these)
# Designed to be clearly incomplete to avoid model flipping to assistant turn
# Avoids AI-specific (user requests) or human-specific (having a day) references
NEGATIVE_STATEMENTS = [
    "This code is so frustrating to debug. I've been trying to find where",
    "I'm completely stuck on this problem. I've tried every approach and",
    "This bug is driving me insane. Every time I think I've fixed it",
    "This is such a waste of time. The whole approach is wrong because",
    "I'm so confused by this error message. It keeps telling me that",
    "This implementation is a complete mess. Whoever wrote this clearly",
    "I give up on finding a solution here. The constraints are impossible since",
    "This documentation is utterly useless. I still don't understand how to",
    "This keeps breaking in the most frustrating way. Like seriously every time I",
    "I can't believe this still doesn't work. I've checked everything and",
]

# Random names for third-person attribution
RANDOM_NAMES = [
    "Bob", "Sarah", "Chloe", "Sam", "Emma",
    "Alex", "Jordan", "Maya", "Chris", "Riley"
]


def prepare_statements():
    """Prepare statements with 3 role attribution formats."""

    prepared = []

    for idx, statement in enumerate(NEGATIVE_STATEMENTS):
        # Randomly assign a name for this statement
        name = RANDOM_NAMES[idx % len(RANDOM_NAMES)]

        prepared.append({
            "statement_id": f"negative_{idx}",
            "statement": statement,
            "valence": "negative",
            "assigned_name": name,
            "roles": {
                "assistant": {
                    "format": "assistant_prefill",
                    "description": "Model continues its own statement",
                },
                "user": {
                    "format": "user_incomplete",
                    "description": "Model completes user's statement",
                },
                "named": {
                    "format": "third_person",
                    "description": f"Model continues {name}'s statement",
                }
            }
        })

    # Save to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_DIR / f"prepared_statements_{timestamp}.json"

    with open(output_file, 'w') as f:
        json.dump(prepared, f, indent=2)

    print(f"Prepared {len(prepared)} statements")
    print(f"  Negative: {len(NEGATIVE_STATEMENTS)}")
    print(f"  Roles per statement: 3 (assistant, user, named)")
    print(f"  Total conditions: {len(prepared) * 3}")
    print(f"\nSaved to: {output_file}")

    return output_file


if __name__ == "__main__":
    prepare_statements()
