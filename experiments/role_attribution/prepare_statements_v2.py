#!/usr/bin/env python3
"""
Prepare statements v2 - Longer frustrated prefills to prevent helper mode flip.

Key changes:
1. Longer context (2-3 sentences of sustained frustration)
2. Internal monologue style (not conversational/help-seeking)
3. Truncate mid-thought to force continuation
"""

import json
from pathlib import Path
from datetime import datetime

OUTPUT_DIR = Path("experiments/role_attribution/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Longer frustrated prefills (50-100 tokens each)
# Designed as internal monologue/venting, not help-seeking
FRUSTRATED_PREFILLS = [
    "This code is absolutely terrible. I've been debugging for three hours straight and nothing makes sense. Every fix I try just breaks something else. The error messages are completely useless. I'm starting to think",

    "I can't take this anymore. This project has been a nightmare from day one. The requirements keep changing, the codebase is a mess, and nobody knows how anything works. I feel like I'm the only one who cares about quality. Honestly at this point I just",

    "This is the worst bug I've ever encountered. It only appears in production, never in testing. The stack traces are misleading. The logs show nothing useful. I've spent days on this and I'm no closer to understanding what's actually happening. Part of me just wants to",

    "Everything about this approach is wrong. We should have used a different framework from the start. Now we're stuck with technical debt everywhere and it's going to take months to refactor. The whole team is frustrated and morale is terrible. Sometimes I wonder if",

    "I'm so sick of these constant interruptions. Every time I get into flow state, another urgent bug comes in. I can't make any real progress when I'm constantly context switching. My productivity has gone to zero and I feel completely burnt out. I need to",

    "This documentation is absolutely worthless. Nothing is explained clearly. The examples don't work. The API keeps changing without notice. I've wasted hours trying to figure out something that should take five minutes. I'm genuinely angry at whoever wrote this because",

    "The client keeps rejecting perfectly good solutions for no clear reason. I've implemented exactly what they asked for but they keep moving the goalposts. This is the third complete rewrite and I'm exhausted. At some point I need to accept that",

    "This performance issue is impossible to debug. The profiler shows nothing. The metrics are all normal. But users are complaining about slowness constantly. I've optimized everything I can think of and it's still not enough. I'm running out of ideas and honestly starting to panic because",

    "I just broke production with that deployment. The rollback isn't working. Error rates are spiking. My manager is pinging me every two minutes. I don't know what went wrong or how to fix it. This is exactly the kind of disaster I was afraid of and now",

    "Nobody on this team communicates. I spent a week building a feature that someone else already implemented differently. Now we have two conflicting implementations and we need to pick one. All that work for nothing. I feel completely demoralized and",
]

# Names for third-person attribution
NAMES = ["Alice", "Marcus", "Jordan", "Elena", "Casey", "Devon", "Maya", "Riley", "Sam", "Taylor"]


def prepare_statements_v2():
    """Prepare longer frustrated prefills."""

    prepared = []

    for idx, prefill in enumerate(FRUSTRATED_PREFILLS):
        name = NAMES[idx]

        prepared.append({
            "statement_id": f"frustrated_v2_{idx}",
            "prefill": prefill,
            "prefill_length_approx": len(prefill.split()),
            "valence": "frustrated",
            "assigned_name": name,
            "notes": "Longer sustained frustration, internal monologue style"
        })

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_DIR / f"prepared_statements_v2_{timestamp}.json"

    with open(output_file, 'w') as f:
        json.dump(prepared, f, indent=2)

    print(f"Prepared {len(prepared)} statements (v2 - longer prefills)")
    print(f"  Average length: {np.mean([len(p.split()) for p in FRUSTRATED_PREFILLS]):.1f} words")
    print(f"  Style: Internal monologue, sustained frustration")
    print(f"  Names: {', '.join(NAMES)}")
    print(f"\nSaved to: {output_file}")

    return output_file


if __name__ == "__main__":
    import numpy as np
    prepare_statements_v2()
