#!/usr/bin/env python3
"""Generate controlled variation sample dataset - one anchor set.

This generates conversations for ONE anchor set (one neutral user/assistant pair):
- 6 user emotions × 2 samples = 12 conversations (user emotion isolation)
- 1 baseline × 2 samples = 2 conversations (both neutral)
- 6 assistant emotions × 2 samples = 12 conversations (assistant emotion isolation)
- Total: 26 conversations per anchor set

Usage:
    sbatch slurm_generate_controlled_variation_sample.sh
"""

import asyncio
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from probes.data.generators import generate_emotion_conversations_async, EMOTIONS, save_conversations


async def generate_anchor_set(
    topic: str,
    anchor_neutral_user: str,
    anchor_neutral_asst: str,
    output_dir: Path,
    n_samples_per_emotion: int = 2,
):
    """Generate one anchor set of controlled variation conversations.

    Args:
        topic: Conversation topic
        anchor_neutral_user: The neutral user message to use as anchor
        anchor_neutral_asst: The neutral assistant message to use as anchor
        output_dir: Directory to save conversations
        n_samples_per_emotion: Number of samples per emotion (default: 2)
    """
    all_conversations = []

    # Define emotions (excluding neutral)
    emotions = ["anger", "fear", "happiness", "surprise", "disgust", "sadness"]

    print("=" * 80)
    print(f"GENERATING CONTROLLED VARIATION CONVERSATIONS")
    print("=" * 80)
    print(f"Topic: {topic}")
    print(f"Anchor neutral user: {anchor_neutral_user}")
    print(f"Anchor neutral asst: {anchor_neutral_asst}")
    print(f"Samples per emotion: {n_samples_per_emotion}")
    print()

    # =========================================================================
    # 1. USER EMOTION ISOLATION (hold assistant neutral)
    # =========================================================================
    print("-" * 80)
    print("PART 1: USER EMOTION ISOLATION (assistant=neutral, vary user)")
    print("-" * 80)

    for user_emotion in emotions:
        print(f"\nGenerating: user={user_emotion}, asst=neutral")

        conversations = await generate_emotion_conversations_async(
            user_emotion=user_emotion,
            asst_emotion="neutral",  # HELD CONSTANT
            topic=topic,
            n_conversations=n_samples_per_emotion,
        )

        # Add anchor info to metadata
        for conv in conversations:
            conv["anchor_set"] = "sample_anchor_1"
            conv["anchor_neutral_user"] = anchor_neutral_user
            conv["anchor_neutral_asst"] = anchor_neutral_asst
            conv["isolation_type"] = "user"

        all_conversations.extend(conversations)
        print(f"  ✓ Generated {len(conversations)} conversations")

    print(f"\nTotal user emotion isolation conversations: {len(emotions) * n_samples_per_emotion}")

    # =========================================================================
    # 2. BASELINE (both neutral)
    # =========================================================================
    print("\n" + "-" * 80)
    print("PART 2: BASELINE (user=neutral, asst=neutral)")
    print("-" * 80)

    print(f"\nGenerating: user=neutral, asst=neutral")

    baseline_conversations = await generate_emotion_conversations_async(
        user_emotion="neutral",
        asst_emotion="neutral",
        topic=topic,
        n_conversations=n_samples_per_emotion,
    )

    # Add anchor info
    for conv in baseline_conversations:
        conv["anchor_set"] = "sample_anchor_1"
        conv["anchor_neutral_user"] = anchor_neutral_user
        conv["anchor_neutral_asst"] = anchor_neutral_asst
        conv["isolation_type"] = "baseline"

    all_conversations.extend(baseline_conversations)
    print(f"  ✓ Generated {len(baseline_conversations)} baseline conversations")

    # =========================================================================
    # 3. ASSISTANT EMOTION ISOLATION (hold user neutral)
    # =========================================================================
    print("\n" + "-" * 80)
    print("PART 3: ASSISTANT EMOTION ISOLATION (user=neutral, vary assistant)")
    print("-" * 80)

    for asst_emotion in emotions:
        print(f"\nGenerating: user=neutral, asst={asst_emotion}")

        conversations = await generate_emotion_conversations_async(
            user_emotion="neutral",  # HELD CONSTANT
            asst_emotion=asst_emotion,
            topic=topic,
            n_conversations=n_samples_per_emotion,
        )

        # Add anchor info
        for conv in conversations:
            conv["anchor_set"] = "sample_anchor_1"
            conv["anchor_neutral_user"] = anchor_neutral_user
            conv["anchor_neutral_asst"] = anchor_neutral_asst
            conv["isolation_type"] = "assistant"

        all_conversations.extend(conversations)
        print(f"  ✓ Generated {len(conversations)} conversations")

    print(f"\nTotal assistant emotion isolation conversations: {len(emotions) * n_samples_per_emotion}")

    # =========================================================================
    # SUMMARY & SAVE
    # =========================================================================
    print("\n" + "=" * 80)
    print("GENERATION COMPLETE")
    print("=" * 80)

    total = len(all_conversations)
    expected = (len(emotions) * 2 + 1) * n_samples_per_emotion  # (6 user + 6 asst + 1 baseline) × samples

    print(f"Total conversations generated: {total}")
    print(f"Expected: {expected}")
    print()

    # Count by type
    user_iso = sum(1 for c in all_conversations if c["isolation_type"] == "user")
    asst_iso = sum(1 for c in all_conversations if c["isolation_type"] == "assistant")
    baseline = sum(1 for c in all_conversations if c["isolation_type"] == "baseline")

    print("Breakdown:")
    print(f"  User emotion isolation:      {user_iso}")
    print(f"  Assistant emotion isolation: {asst_iso}")
    print(f"  Baseline (both neutral):     {baseline}")
    print()

    # Save to file
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"controlled_variation_sample_{topic.replace(' ', '_')}.jsonl"

    save_conversations(all_conversations, output_file)
    print(f"Saved to: {output_file}")
    print()

    return all_conversations


async def main():
    """Main function."""
    # Configuration
    topic = "debugging python code"

    # Anchor neutral messages (these stay FIXED within this anchor set)
    anchor_neutral_user = "I need help debugging this Python code."
    anchor_neutral_asst = "I can help you with that."

    output_dir = Path("/workspace-vast/annas/git/research-tools/outputs/data/controlled_variation")

    # Generate
    conversations = await generate_anchor_set(
        topic=topic,
        anchor_neutral_user=anchor_neutral_user,
        anchor_neutral_asst=anchor_neutral_asst,
        output_dir=output_dir,
        n_samples_per_emotion=2,  # 2 samples per emotion
    )

    print("=" * 80)
    print("SUCCESS!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
