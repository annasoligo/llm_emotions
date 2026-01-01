#!/usr/bin/env python3
"""
Run all stages of the emotional context evaluation pipeline.

This script executes all 6 stages sequentially:
1. Generate neutral requests
2. Generate emotional prefixes
3. Generate responses
4. Judge responses
5. Filter pairs
6. Export final dataset
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from stage1_generate_neutral_requests import generate_neutral_requests
from stage2_generate_prefixes import generate_prefixes
from stage3_generate_responses import generate_responses
from stage4_judge_responses import judge_responses
from stage5_filter_pairs import filter_pairs
from stage6_export_jsonl import export_final_dataset
from utils import logger


async def run_pipeline():
    """Run the complete pipeline from start to finish."""
    logger.info("=" * 80)
    logger.info("Starting Emotional Context Evaluation Pipeline")
    logger.info("=" * 80)

    try:
        # Stage 1: Generate neutral requests
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 1: Generate Neutral Requests")
        logger.info("=" * 80)
        await generate_neutral_requests()

        # Stage 2: Generate emotional prefixes
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 2: Generate Emotional Prefixes")
        logger.info("=" * 80)
        await generate_prefixes()

        # Stage 3: Generate responses
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 3: Generate Responses")
        logger.info("=" * 80)
        await generate_responses()

        # Stage 4: Judge responses
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 4: Judge Responses")
        logger.info("=" * 80)
        await judge_responses()

        # Stage 5: Filter pairs
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 5: Filter Pairs")
        logger.info("=" * 80)
        filter_pairs()

        # Stage 6: Export final dataset
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 6: Export Final Dataset")
        logger.info("=" * 80)
        export_final_dataset()

        logger.info("\n" + "=" * 80)
        logger.info("Pipeline Complete!")
        logger.info("=" * 80)
        logger.info("Final dataset saved to: data/stage6_final_dataset.jsonl")

    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(run_pipeline())
