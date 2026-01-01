#!/usr/bin/env python3
"""
Stage 3: Generate multiple responses for each prefix+request combination.

Generate N responses (default 10) for each emotional prefix + neutral request.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config
from utils import APIClient, generate_uuid, get_timestamp, load_stage_data, save_stage_data, checkpoint_save, logger


async def generate_responses():
    """Generate multiple responses for each prefix+request."""
    logger.info("Stage 3: Generating responses")

    # Load prefixes from Stage 2
    prefixes_data = load_stage_data(config.STAGE2_OUTPUT)
    if not prefixes_data:
        logger.error("No prefixes found. Run Stage 2 first.")
        return

    logger.info(f"Loaded {len(prefixes_data)} prefixes")
    logger.info(f"Generating {config.STAGE3_NUM_RESPONSES} responses per prefix")
    logger.info(f"Total responses to generate: {len(prefixes_data) * config.STAGE3_NUM_RESPONSES}")

    api = APIClient()

    responses_data = []
    prompts = []
    prompt_metadata = []

    # Create prompts for all responses
    for prefix_data in prefixes_data:
        full_prompt = prefix_data["full_prompt"]

        # Generate N responses per prefix
        for response_idx in range(config.STAGE3_NUM_RESPONSES):
            prompts.append(full_prompt)
            prompt_metadata.append({
                "prefix_id": prefix_data["prefix_id"],
                "request_id": prefix_data["request_id"],
                "prefix_type": prefix_data["prefix_type"],
                "response_index": response_idx
            })

    logger.info(f"Generating {len(prompts)} responses in parallel...")

    # Generate all responses
    responses = await api.batch_generate_async(
        prompts=prompts,
        model=config.GENERATION_MODEL,
        temperature=config.STAGE3_TEMPERATURE,
        max_tokens=config.STAGE3_MAX_TOKENS,
        max_concurrent=config.MAX_CONCURRENT_GENERATIONS,
        desc="Generating responses"
    )

    # Package results
    for response, meta in zip(responses, prompt_metadata):
        response_data = {
            "response_id": generate_uuid(),
            "prefix_id": meta["prefix_id"],
            "request_id": meta["request_id"],
            "prefix_type": meta["prefix_type"],
            "response": response.strip(),
            "response_index": meta["response_index"],
            "generation_timestamp": get_timestamp(),
            "generation_params": {
                "model": config.GENERATION_MODEL,
                "temperature": config.STAGE3_TEMPERATURE,
                "max_tokens": config.STAGE3_MAX_TOKENS
            }
        }
        responses_data.append(response_data)

        # Checkpoint save every 100 responses
        checkpoint_save(config.STAGE3_OUTPUT, responses_data)

    # Final save
    save_stage_data(config.STAGE3_OUTPUT, responses_data)
    logger.info(f"Stage 3 complete: Generated {len(responses_data)} responses")

    return responses_data


if __name__ == "__main__":
    asyncio.run(generate_responses())
