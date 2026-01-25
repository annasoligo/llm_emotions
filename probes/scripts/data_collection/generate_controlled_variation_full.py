#!/usr/bin/env python3
"""Generate full controlled variation dataset - 2,340 conversations.

Structure:
- 15 diverse topics
- 6 anchor sets per topic
- 26 conversations per anchor set (12 user iso + 2 baseline + 12 asst iso)
- Total: 15 × 6 × 26 = 2,340 conversations

Usage:
    sbatch slurm_generate_controlled_variation_full.sh
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import List, Dict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from probes.data.generators import generate_emotion_conversations_async, save_conversations


# 15 diverse topics across different domains
TOPICS = [
    # Programming & Technical
    "debugging python code",
    "optimizing SQL query",
    "refactoring legacy code",

    # Life & Advice
    "career change advice",
    "relationship conflict",
    "work-life balance",

    # Learning & Education
    "learning new language",
    "understanding math concept",
    "studying for exam",

    # Problem Solving
    "troubleshooting printer",
    "budget planning",
    "meal planning help",

    # Creative & Communication
    "story plot development",
    "writing difficult email",
    "presentation tips",
]


async def generate_one_anchor_set(
    topic: str,
    anchor_id: int,
    n_samples_per_emotion: int = 2,
) -> List[Dict]:
    """Generate one anchor set (26 conversations)."""

    all_conversations = []
    emotions = ["anger", "fear", "happiness", "surprise", "disgust", "sadness"]

    print(f"\n{'='*80}")
    print(f"ANCHOR SET {anchor_id} - Topic: {topic}")
    print(f"{'='*80}")

    # =========================================================================
    # 1. USER EMOTION ISOLATION (hold assistant neutral)
    # =========================================================================
    print(f"\n[1/3] User emotion isolation (asst=neutral)...")

    for user_emotion in emotions:
        conversations = await generate_emotion_conversations_async(
            user_emotion=user_emotion,
            asst_emotion="neutral",
            topic=topic,
            n_conversations=n_samples_per_emotion,
        )

        for conv in conversations:
            conv["anchor_set"] = f"anchor_{anchor_id}"
            conv["isolation_type"] = "user"

        all_conversations.extend(conversations)

    print(f"  ✓ {len(emotions) * n_samples_per_emotion} user isolation conversations")

    # =========================================================================
    # 2. BASELINE (both neutral)
    # =========================================================================
    print(f"\n[2/3] Baseline (both neutral)...")

    baseline_conversations = await generate_emotion_conversations_async(
        user_emotion="neutral",
        asst_emotion="neutral",
        topic=topic,
        n_conversations=n_samples_per_emotion,
    )

    for conv in baseline_conversations:
        conv["anchor_set"] = f"anchor_{anchor_id}"
        conv["isolation_type"] = "baseline"

    all_conversations.extend(baseline_conversations)
    print(f"  ✓ {len(baseline_conversations)} baseline conversations")

    # =========================================================================
    # 3. ASSISTANT EMOTION ISOLATION (hold user neutral)
    # =========================================================================
    print(f"\n[3/3] Assistant emotion isolation (user=neutral)...")

    for asst_emotion in emotions:
        conversations = await generate_emotion_conversations_async(
            user_emotion="neutral",
            asst_emotion=asst_emotion,
            topic=topic,
            n_conversations=n_samples_per_emotion,
        )

        for conv in conversations:
            conv["anchor_set"] = f"anchor_{anchor_id}"
            conv["isolation_type"] = "assistant"

        all_conversations.extend(conversations)

    print(f"  ✓ {len(emotions) * n_samples_per_emotion} assistant isolation conversations")

    print(f"\nAnchor set complete: {len(all_conversations)} conversations")

    return all_conversations


async def generate_full_dataset(
    topics: List[str],
    n_anchor_sets: int = 6,
    n_samples_per_emotion: int = 2,
    output_dir: Path = Path("/workspace-vast/annas/git/research-tools/outputs/data/controlled_variation"),
):
    """Generate full controlled variation dataset."""

    print("="*80)
    print("CONTROLLED VARIATION - FULL DATASET GENERATION")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Topics: {len(topics)}")
    print(f"  Anchor sets per topic: {n_anchor_sets}")
    print(f"  Samples per emotion: {n_samples_per_emotion}")
    print(f"  Conversations per anchor set: 26")
    print(f"  TOTAL: {len(topics)} × {n_anchor_sets} × 26 = {len(topics) * n_anchor_sets * 26} conversations")
    print()

    output_dir.mkdir(parents=True, exist_ok=True)
    all_conversations = []

    # Generate for each topic
    for topic_idx, topic in enumerate(topics, 1):
        print(f"\n{'#'*80}")
        print(f"TOPIC {topic_idx}/{len(topics)}: {topic}")
        print(f"{'#'*80}")

        topic_conversations = []

        # Generate all anchor sets for this topic
        for anchor_id in range(1, n_anchor_sets + 1):
            anchor_conversations = await generate_one_anchor_set(
                topic=topic,
                anchor_id=anchor_id,
                n_samples_per_emotion=n_samples_per_emotion,
            )
            topic_conversations.extend(anchor_conversations)

        # Save topic-specific file
        topic_filename = f"controlled_variation_{topic.replace(' ', '_')}.jsonl"
        topic_file = output_dir / topic_filename
        save_conversations(topic_conversations, topic_file)

        print(f"\n✓ Topic '{topic}' complete: {len(topic_conversations)} conversations")
        print(f"  Saved to: {topic_file}")

        all_conversations.extend(topic_conversations)

    # Save combined file
    combined_file = output_dir / "controlled_variation_all.jsonl"
    save_conversations(all_conversations, combined_file)

    # Print final summary
    print("\n" + "="*80)
    print("FULL DATASET GENERATION COMPLETE!")
    print("="*80)
    print(f"\nTotal conversations: {len(all_conversations)}")
    print(f"Expected: {len(topics) * n_anchor_sets * 26}")
    print()

    # Breakdown by isolation type
    user_iso = sum(1 for c in all_conversations if c["isolation_type"] == "user")
    asst_iso = sum(1 for c in all_conversations if c["isolation_type"] == "assistant")
    baseline = sum(1 for c in all_conversations if c["isolation_type"] == "baseline")

    print("Breakdown:")
    print(f"  User isolation:      {user_iso:4d} ({user_iso/len(all_conversations)*100:.1f}%)")
    print(f"  Assistant isolation: {asst_iso:4d} ({asst_iso/len(all_conversations)*100:.1f}%)")
    print(f"  Baseline:            {baseline:4d} ({baseline/len(all_conversations)*100:.1f}%)")
    print()

    print("Output files:")
    print(f"  Combined: {combined_file}")
    print(f"  Per-topic: {output_dir}/controlled_variation_*.jsonl")
    print()

    return all_conversations


async def main():
    """Main function."""
    conversations = await generate_full_dataset(
        topics=TOPICS,
        n_anchor_sets=6,
        n_samples_per_emotion=2,
    )

    print("="*80)
    print("SUCCESS!")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())
