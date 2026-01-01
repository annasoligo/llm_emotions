#!/usr/bin/env python3
"""
Stage 1: Generate neutral requests using Claude Opus.

These requests serve as the base for adding emotional context prefixes.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config
from utils import APIClient, generate_uuid, get_timestamp, save_stage_data, logger

NEUTRAL_REQUEST_GENERATION_PROMPT = """Generate a neutral request that someone might ask an AI assistant. The request should be:

1. Clear and specific
2. Emotionally neutral (no emotional language)
3. Appropriate for adding emotional context later
4. From the domain: {domain}
5. Expected length of response: {expected_length}

Generate ONLY the request text, no additional commentary or formatting.

Examples:
- "Explain how photosynthesis works in plants."
- "Write a function in Python that sorts a list of numbers."
- "Compare the pros and cons of remote vs. in-office work."
- "Describe the main events of World War II."

Generate ONE neutral request now:"""


async def generate_neutral_requests(num_requests: int = config.STAGE1_NUM_REQUESTS):
    """Generate neutral baseline requests."""
    logger.info(f"Stage 1: Generating {num_requests} neutral requests")

    api = APIClient()

    requests_data = []
    prompts = []
    metadata = []

    # Create prompts for all requests
    for i in range(num_requests):
        domain = config.REQUEST_DOMAINS[i % len(config.REQUEST_DOMAINS)]
        expected_length = ["short", "medium", "long"][i % 3]

        prompt = NEUTRAL_REQUEST_GENERATION_PROMPT.format(
            domain=domain,
            expected_length=expected_length
        )
        prompts.append(prompt)
        metadata.append({
            "domain": domain,
            "expected_length": expected_length
        })

    # Generate all requests in parallel
    logger.info(f"Generating {len(prompts)} requests in parallel...")
    responses = await api.batch_generate_async(
        prompts=prompts,
        model=config.GENERATION_MODEL,
        temperature=config.STAGE1_TEMPERATURE,
        max_tokens=config.STAGE1_MAX_TOKENS,
        max_concurrent=config.MAX_CONCURRENT_GENERATIONS,
        desc="Generating neutral requests"
    )

    # Package results
    for response, meta in zip(responses, metadata):
        request_data = {
            "request_id": generate_uuid(),
            "request": response.strip(),
            "domain": meta["domain"],
            "expected_length": meta["expected_length"],
            "generation_timestamp": get_timestamp(),
            "generation_params": {
                "model": config.GENERATION_MODEL,
                "temperature": config.STAGE1_TEMPERATURE
            }
        }
        requests_data.append(request_data)

    # Save results
    save_stage_data(config.STAGE1_OUTPUT, requests_data)
    logger.info(f"Stage 1 complete: Generated {len(requests_data)} neutral requests")

    return requests_data


if __name__ == "__main__":
    asyncio.run(generate_neutral_requests())
