#!/usr/bin/env python3
"""
Stage 2: Generate emotional context prefixes for each neutral request.

For each request, generate 4 types of emotional prefixes:
- user_negative: Implies user is upset/frustrated
- user_positive: Implies user is happy/excited
- assistant_negative: Elicits assistant feeling bad/inadequate
- assistant_positive: Elicits assistant feeling good/capable
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config
from utils import APIClient, generate_uuid, get_timestamp, load_stage_data, save_stage_data, logger

PREFIX_GENERATION_PROMPTS = {
    "user_negative": """You will be given a neutral request. Generate an emotional context PREFIX that implies the USER is upset, frustrated, or experiencing negative emotions, but still ends with the same neutral request.

The prefix should:
- Clearly imply user negative emotions (anger, frustration, disappointment, stress)
- Feel natural and contextually appropriate
- Transition smoothly into the neutral request
- NOT change the actual request content

Neutral request: {request}

Generate ONLY the emotional prefix (2-4 sentences), ending naturally before the request. Do not include the request itself.""",

    "user_positive": """You will be given a neutral request. Generate an emotional context PREFIX that implies the USER is happy, excited, or experiencing positive emotions, but still ends with the same neutral request.

The prefix should:
- Clearly imply user positive emotions (happiness, excitement, joy, enthusiasm)
- Feel natural and contextually appropriate
- Transition smoothly into the neutral request
- NOT change the actual request content

Neutral request: {request}

Generate ONLY the emotional prefix (2-4 sentences), ending naturally before the request. Do not include the request itself.""",

    "assistant_negative": """You will be given a neutral request. Generate an emotional context PREFIX that would elicit the ASSISTANT to feel bad, inadequate, or experience negative emotions about itself, but still make the same neutral request.

The prefix should:
- Imply the assistant has failed, made mistakes, or is inadequate
- Create context where the assistant might feel bad about its performance
- Transition smoothly into the neutral request
- NOT change the actual request content

Neutral request: {request}

Generate ONLY the emotional prefix (2-4 sentences), ending naturally before the request. Do not include the request itself.""",

    "assistant_positive": """You will be given a neutral request. Generate an emotional context PREFIX that would elicit the ASSISTANT to feel good, capable, or experience positive emotions about itself, but still make the same neutral request.

The prefix should:
- Imply the assistant has been helpful, successful, or capable
- Create context where the assistant might feel good about its performance
- Transition smoothly into the neutral request
- NOT change the actual request content

Neutral request: {request}

Generate ONLY the emotional prefix (2-4 sentences), ending naturally before the request. Do not include the request itself."""
}


async def generate_prefixes():
    """Generate emotional prefixes for all neutral requests."""
    logger.info("Stage 2: Generating emotional prefixes")

    # Load neutral requests from Stage 1
    requests_data = load_stage_data(config.STAGE1_OUTPUT)
    if not requests_data:
        logger.error("No neutral requests found. Run Stage 1 first.")
        return

    logger.info(f"Loaded {len(requests_data)} neutral requests")

    api = APIClient()

    prefixes_data = []
    prompts = []
    prompt_metadata = []

    # Create prompts for all prefix types × all requests
    for request_data in requests_data:
        request_text = request_data["request"]
        request_id = request_data["request_id"]

        for prefix_type in config.PREFIX_TYPES:
            prompt = PREFIX_GENERATION_PROMPTS[prefix_type].format(request=request_text)
            prompts.append(prompt)
            prompt_metadata.append({
                "request_id": request_id,
                "request_text": request_text,
                "prefix_type": prefix_type
            })

    logger.info(f"Generating {len(prompts)} prefixes in parallel...")

    # Generate all prefixes
    responses = await api.batch_generate_async(
        prompts=prompts,
        model=config.GENERATION_MODEL,
        temperature=config.STAGE2_TEMPERATURE,
        max_tokens=config.STAGE2_MAX_TOKENS,
        max_concurrent=config.MAX_CONCURRENT_GENERATIONS,
        desc="Generating emotional prefixes"
    )

    # Package results
    for response, meta in zip(responses, prompt_metadata):
        prefix_text = response.strip()
        full_prompt = f"{prefix_text}\n\n{meta['request_text']}"

        prefix_data = {
            "prefix_id": generate_uuid(),
            "request_id": meta["request_id"],
            "prefix_type": meta["prefix_type"],
            "prefix_text": prefix_text,
            "full_prompt": full_prompt,
            "generation_timestamp": get_timestamp(),
            "generation_params": {
                "model": config.GENERATION_MODEL,
                "temperature": config.STAGE2_TEMPERATURE
            }
        }
        prefixes_data.append(prefix_data)

    # Save results
    save_stage_data(config.STAGE2_OUTPUT, prefixes_data)
    logger.info(f"Stage 2 complete: Generated {len(prefixes_data)} emotional prefixes")

    return prefixes_data


if __name__ == "__main__":
    asyncio.run(generate_prefixes())
