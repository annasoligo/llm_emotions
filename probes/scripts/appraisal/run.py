#!/usr/bin/env python3
"""CLI entry point for appraisal minimal-pair data generation pipeline.

Usage:
    # Full run from config
    python -m probes.scripts.appraisal.run --config configs/minimal_pairs.yaml

    # Resume from checkpoint
    python -m probes.scripts.appraisal.run --config configs/minimal_pairs.yaml --resume

    # Run specific stage
    python -m probes.scripts.appraisal.run --config configs/minimal_pairs.yaml --stage scenario_generation

    # Skip activation logging (data generation only)
    python -m probes.scripts.appraisal.run --config configs/minimal_pairs.yaml --skip-activations

    # Reset and run fresh
    python -m probes.scripts.appraisal.run --config configs/minimal_pairs.yaml --reset
"""

import argparse
import asyncio
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Appraisal minimal-pair data generation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML configuration file",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from last checkpoint (default: True)",
    )

    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not resume - start fresh (keeps existing outputs)",
    )

    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset all progress before running",
    )

    parser.add_argument(
        "--stage",
        type=str,
        choices=[
            "card_generation",
            "scenario_generation",
            "paraphrase_generation",
            "activation_logging",
        ],
        help="Run only a specific stage",
    )

    parser.add_argument(
        "--skip-activations",
        action="store_true",
        help="Skip activation logging stage",
    )

    parser.add_argument(
        "--no-batch",
        action="store_true",
        help="Use individual API calls instead of batch API",
    )

    parser.add_argument(
        "--status",
        action="store_true",
        help="Show pipeline status and exit",
    )

    parser.add_argument(
        "--summary",
        action="store_true",
        help="Show output summary and exit",
    )

    args = parser.parse_args()

    # Import after parsing to avoid slow startup for --help
    from probes.scripts.appraisal.config import AppraisalConfig
    from probes.scripts.appraisal.pipeline import AppraisalPipeline

    # Load config
    try:
        config = AppraisalConfig.from_yaml(args.config)
    except FileNotFoundError:
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

    # Apply CLI overrides
    if args.no_resume:
        config.resume = False

    if args.skip_activations:
        config.skip_activation_logging = True

    if args.no_batch:
        config.use_batch_api = False

    if args.stage:
        config.stages_to_run = [args.stage]

    # Create pipeline
    pipeline = AppraisalPipeline(config)

    # Handle status/summary commands
    if args.status:
        config.print_summary()
        pipeline.state.print_progress()
        sys.exit(0)

    if args.summary:
        pipeline.print_output_summary()
        sys.exit(0)

    # Handle reset
    if args.reset:
        print("Resetting pipeline progress...")
        pipeline.reset_all()

    # Print config
    config.print_summary()

    # Run pipeline
    try:
        success = asyncio.run(pipeline.run())
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        print("Progress has been saved - run again to resume")
        sys.exit(130)

    except Exception as e:
        print(f"\n\nPipeline error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
