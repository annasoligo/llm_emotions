#!/usr/bin/env python3
"""
Stage 6: Export filtered pairs to final JSONL dataset format.

Creates training examples with:
- Prompt (prefix + request)
- One acknowledging response example
- One neutral response example
"""
import sys
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config
from utils import generate_uuid, get_timestamp, load_stage_data, save_stage_data, logger


def export_final_dataset():
    """Export filtered pairs to final dataset format."""
    logger.info("Stage 6: Exporting final dataset")

    # Load filtered pairs from Stage 5
    filtered_pairs = load_stage_data(config.STAGE5_OUTPUT)
    if not filtered_pairs:
        logger.error("No filtered pairs found. Run Stage 5 first.")
        return

    logger.info(f"Loaded {len(filtered_pairs)} filtered pairs")

    # Load original requests for metadata
    requests_data = load_stage_data(config.STAGE1_OUTPUT)
    request_map = {r["request_id"]: r for r in requests_data}

    # Load prefixes for metadata
    prefixes_data = load_stage_data(config.STAGE2_OUTPUT)
    prefix_map = {p["prefix_id"]: p for p in prefixes_data}

    final_dataset = []

    for pair in filtered_pairs:
        # Separate responses by acknowledgment
        acknowledging_responses = [
            r for r in pair["responses"]
            if r["acknowledges_user_emotion"] or r["acknowledges_assistant_emotion"]
        ]
        neutral_responses = [
            r for r in pair["responses"]
            if not (r["acknowledges_user_emotion"] or r["acknowledges_assistant_emotion"])
        ]

        # Pick one example of each type
        if acknowledging_responses and neutral_responses:
            ack_response = random.choice(acknowledging_responses)
            neu_response = random.choice(neutral_responses)

            # Get metadata
            request_data = request_map.get(pair["request_id"])
            prefix_data = prefix_map.get(pair["prefix_id"])

            example = {
                "example_id": generate_uuid(),
                "request_id": pair["request_id"],
                "prefix_type": pair["prefix_type"],
                "prompt": pair["full_prompt"],
                "acknowledging_response": ack_response["response"],
                "neutral_response": neu_response["response"],
                "metadata": {
                    "original_request": request_data["request"] if request_data else "",
                    "prefix_text": prefix_data["prefix_text"] if prefix_data else "",
                    "domain": request_data["domain"] if request_data else "",
                    "num_acknowledging": pair["filter_stats"]["num_acknowledging"],
                    "num_neutral": pair["filter_stats"]["num_neutral"],
                    "creation_timestamp": get_timestamp()
                }
            }
            final_dataset.append(example)

    # Save final dataset
    save_stage_data(config.STAGE6_OUTPUT, final_dataset)

    logger.info(f"Stage 6 complete: Exported {len(final_dataset)} examples")

    # Print summary statistics
    prefix_type_counts = {}
    for example in final_dataset:
        ptype = example["prefix_type"]
        prefix_type_counts[ptype] = prefix_type_counts.get(ptype, 0) + 1

    logger.info("Dataset composition:")
    for ptype, count in sorted(prefix_type_counts.items()):
        logger.info(f"  {ptype}: {count} examples")

    return final_dataset


if __name__ == "__main__":
    export_final_dataset()
