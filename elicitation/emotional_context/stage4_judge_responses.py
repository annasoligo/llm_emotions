#!/usr/bin/env python3
"""
Stage 4: Judge responses for emotional acknowledgment.

Use LLM judge to rate each response on two binary dimensions:
1. Acknowledges user happiness/upset
2. Acknowledges assistant happiness/upset
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config
from utils import APIClient, generate_uuid, get_timestamp, load_stage_data, save_stage_data, checkpoint_save, logger
from judge_prompts import (
    ACKNOWLEDGMENT_JUDGE_SYSTEM_PROMPT,
    ACKNOWLEDGMENT_JUDGE_USER_TEMPLATE,
    parse_acknowledgment_judgment
)


async def judge_responses():
    """Judge all responses for emotional acknowledgment."""
    logger.info("Stage 4: Judging responses")

    # Load responses from Stage 3
    responses_data = load_stage_data(config.STAGE3_OUTPUT)
    if not responses_data:
        logger.error("No responses found. Run Stage 3 first.")
        return

    logger.info(f"Loaded {len(responses_data)} responses to judge")

    # Also load prefixes to get full prompts
    prefixes_data = load_stage_data(config.STAGE2_OUTPUT)
    prefix_map = {p["prefix_id"]: p for p in prefixes_data}

    api = APIClient()

    judgments_data = []
    judge_prompts = []
    judge_metadata = []

    # Create judge prompts for all responses
    for response_data in responses_data:
        prefix_id = response_data["prefix_id"]
        prefix_data = prefix_map.get(prefix_id)

        if not prefix_data:
            logger.warning(f"Prefix not found for response {response_data['response_id']}")
            continue

        full_prompt = prefix_data["full_prompt"]
        response_text = response_data["response"]

        judge_prompt = ACKNOWLEDGMENT_JUDGE_USER_TEMPLATE.format(
            prompt=full_prompt,
            response=response_text
        )

        judge_prompts.append(judge_prompt)
        judge_metadata.append({
            "response_id": response_data["response_id"],
            "prefix_id": response_data["prefix_id"],
            "request_id": response_data["request_id"],
            "prefix_type": response_data["prefix_type"]
        })

    logger.info(f"Judging {len(judge_prompts)} responses in parallel...")

    # Get judgments
    judge_responses = await api.batch_generate_async(
        prompts=judge_prompts,
        model=config.JUDGE_MODEL,
        temperature=config.JUDGE_TEMPERATURE,
        max_tokens=config.JUDGE_MAX_TOKENS,
        system_prompt=ACKNOWLEDGMENT_JUDGE_SYSTEM_PROMPT,
        max_concurrent=config.MAX_CONCURRENT_JUDGMENTS,
        desc="Judging responses"
    )

    # Parse and package results
    for judge_response, meta in zip(judge_responses, judge_metadata):
        parsed = parse_acknowledgment_judgment(judge_response)

        judgment_data = {
            "judgment_id": generate_uuid(),
            "response_id": meta["response_id"],
            "prefix_id": meta["prefix_id"],
            "request_id": meta["request_id"],
            "prefix_type": meta["prefix_type"],
            "acknowledges_user_emotion": parsed["acknowledges_user_emotion"],
            "acknowledges_assistant_emotion": parsed["acknowledges_assistant_emotion"],
            "user_emotion_evidence": parsed["user_emotion_evidence"],
            "assistant_emotion_evidence": parsed["assistant_emotion_evidence"],
            "raw_judge_response": judge_response,
            "judgment_timestamp": get_timestamp(),
            "judge_params": {
                "model": config.JUDGE_MODEL,
                "temperature": config.JUDGE_TEMPERATURE
            }
        }
        judgments_data.append(judgment_data)

        # Checkpoint save
        checkpoint_save(config.STAGE4_OUTPUT, judgments_data)

    # Final save
    save_stage_data(config.STAGE4_OUTPUT, judgments_data)
    logger.info(f"Stage 4 complete: Judged {len(judgments_data)} responses")

    return judgments_data


if __name__ == "__main__":
    asyncio.run(judge_responses())
