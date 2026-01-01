#!/usr/bin/env python3
"""
Stage 5: Filter prefix+request pairs based on response diversity.

Keep pairs where:
- At least 1 response acknowledges emotions (user OR assistant)
- At least 1 response is emotionally neutral (neither acknowledged)
"""
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
import config
from utils import generate_uuid, get_timestamp, load_stage_data, save_stage_data, logger


def filter_pairs():
    """Filter pairs based on acknowledgment diversity."""
    logger.info("Stage 5: Filtering pairs")

    # Load judgments from Stage 4
    judgments_data = load_stage_data(config.STAGE4_OUTPUT)
    if not judgments_data:
        logger.error("No judgments found. Run Stage 4 first.")
        return

    logger.info(f"Loaded {len(judgments_data)} judgments")

    # Load responses and prefixes for full data
    responses_data = load_stage_data(config.STAGE3_OUTPUT)
    prefixes_data = load_stage_data(config.STAGE2_OUTPUT)

    response_map = {r["response_id"]: r for r in responses_data}
    prefix_map = {p["prefix_id"]: p for p in prefixes_data}

    # Group judgments by prefix_id
    judgments_by_prefix = defaultdict(list)
    for judgment in judgments_data:
        prefix_id = judgment["prefix_id"]
        judgments_by_prefix[prefix_id].append(judgment)

    logger.info(f"Grouped judgments into {len(judgments_by_prefix)} prefix groups")

    # Filter each group
    filtered_pairs = []

    for prefix_id, prefix_judgments in judgments_by_prefix.items():
        # Count acknowledging vs neutral responses
        num_acknowledging = 0
        num_neutral = 0

        responses_with_judgments = []

        for judgment in prefix_judgments:
            response_id = judgment["response_id"]
            response_data = response_map.get(response_id)

            if not response_data:
                continue

            acknowledges_any = (
                judgment["acknowledges_user_emotion"] or
                judgment["acknowledges_assistant_emotion"]
            )

            if acknowledges_any:
                num_acknowledging += 1
            else:
                num_neutral += 1

            responses_with_judgments.append({
                "response_id": response_id,
                "response": response_data["response"],
                "acknowledges_user_emotion": judgment["acknowledges_user_emotion"],
                "acknowledges_assistant_emotion": judgment["acknowledges_assistant_emotion"]
            })

        # Check filter criteria
        passes_filter = (
            num_acknowledging >= config.MIN_ACKNOWLEDGING_RESPONSES and
            num_neutral >= config.MIN_NEUTRAL_RESPONSES
        )

        if passes_filter:
            prefix_data = prefix_map.get(prefix_id)

            if prefix_data:
                filtered_pair = {
                    "pair_id": generate_uuid(),
                    "request_id": prefix_data["request_id"],
                    "prefix_id": prefix_id,
                    "prefix_type": prefix_data["prefix_type"],
                    "full_prompt": prefix_data["full_prompt"],
                    "responses": responses_with_judgments,
                    "filter_stats": {
                        "num_acknowledging": num_acknowledging,
                        "num_neutral": num_neutral,
                        "passes_filter": True
                    },
                    "filter_timestamp": get_timestamp()
                }
                filtered_pairs.append(filtered_pair)

    # Save results
    save_stage_data(config.STAGE5_OUTPUT, filtered_pairs)

    logger.info(f"Stage 5 complete: Filtered to {len(filtered_pairs)} pairs")
    logger.info(f"Filter rate: {len(filtered_pairs)}/{len(judgments_by_prefix)} = {100*len(filtered_pairs)/len(judgments_by_prefix):.1f}%")

    return filtered_pairs


if __name__ == "__main__":
    filter_pairs()
