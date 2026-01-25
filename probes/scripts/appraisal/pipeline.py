"""Pipeline orchestrator for appraisal data generation."""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Type

from probes.scripts.appraisal.config import AppraisalConfig
from probes.scripts.appraisal.utils.progress import StateManager
from probes.scripts.appraisal.stages.base import Stage
from probes.scripts.appraisal.stages.card_generation import CardGenerationStage
from probes.scripts.appraisal.stages.scenario_generation import ScenarioGenerationStage
from probes.scripts.appraisal.stages.paraphrase_generation import ParaphraseGenerationStage
from probes.scripts.appraisal.stages.activation_logging import ActivationLoggingStage


# Stage execution order
STAGE_ORDER = [
    ("card_generation", CardGenerationStage),
    ("scenario_generation", ScenarioGenerationStage),
    ("paraphrase_generation", ParaphraseGenerationStage),
    ("activation_logging", ActivationLoggingStage),
]


class AppraisalPipeline:
    """Orchestrator for appraisal minimal-pair data generation.

    Manages stage execution with:
    - Sequential stage execution
    - Resume support via state manager
    - Batch API mode for LLM stages
    - Progress tracking and reporting

    Usage:
        config = AppraisalConfig.from_yaml("config.yaml")
        pipeline = AppraisalPipeline(config)
        await pipeline.run()
    """

    def __init__(self, config: AppraisalConfig):
        """Initialize pipeline.

        Args:
            config: Pipeline configuration
        """
        self.config = config

        # Ensure output directory exists
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize state manager
        self.state = StateManager(
            output_dir=config.output_dir,
            config=config.to_dict(),
        )

        # Save config copy to output dir
        config_copy_path = config.output_dir / "config.yaml"
        if not config_copy_path.exists():
            config.to_yaml(config_copy_path)

    def _get_stages_to_run(self) -> List[tuple]:
        """Get list of stages to run based on config.

        Returns:
            List of (name, stage_class) tuples
        """
        stages = []

        for name, stage_class in STAGE_ORDER:
            # Skip activation logging if configured
            if name == "activation_logging" and self.config.skip_activation_logging:
                continue

            # Filter by stages_to_run if specified
            if self.config.stages_to_run is not None:
                if name not in self.config.stages_to_run:
                    continue

            stages.append((name, stage_class))

        return stages

    async def run(self) -> bool:
        """Run the full pipeline.

        Executes stages sequentially with resume support.

        Returns:
            True if all stages completed successfully
        """
        import sys
        start_time = datetime.now()

        print("\n" + "=" * 70, flush=True)
        print("APPRAISAL MINIMAL-PAIR DATA GENERATION PIPELINE", flush=True)
        print("=" * 70, flush=True)
        print(f"Run name: {self.config.name}", flush=True)
        print(f"Output directory: {self.config.output_dir}", flush=True)
        print(f"Started: {start_time.isoformat()}", flush=True)
        sys.stdout.flush()

        # Print progress if resuming
        if self.config.resume:
            self.state.print_progress()

        stages_to_run = self._get_stages_to_run()
        print(f"\nStages to run: {[name for name, _ in stages_to_run]}")

        all_success = True

        for stage_name, stage_class in stages_to_run:
            print(f"\n{'='*70}")
            print(f"Running stage: {stage_name}")
            print(f"{'='*70}")

            # Create stage instance
            stage = stage_class(self.config, self.state)

            # Check if input file exists for stages that need it
            if hasattr(stage, 'input_file') and stage.input_file is not None:
                input_path = self.config.get_output_path(stage.input_file)
                if not input_path.exists():
                    print(f"Skipping stage {stage_name}: input file not found ({stage.input_file})")
                    print("Previous stage may have produced no output.")
                    all_success = False
                    continue

            try:
                # Run stage (batch mode for LLM stages, regular for activation)
                if stage_name == "activation_logging":
                    success = await stage.run()
                elif self.config.use_batch_api:
                    success = await stage.run_batch()
                else:
                    success = await stage.run(concurrency=self.config.concurrency)

                if not success:
                    print(f"\nStage {stage_name} completed with errors")
                    all_success = False
                    # Stop pipeline if stage had no successes
                    if hasattr(stage, 'output_path') and not stage.output_path.exists():
                        print("Stage produced no output - stopping pipeline")
                        break

            except Exception as e:
                print(f"\nStage {stage_name} failed: {e}")
                all_success = False
                break

        # Final summary
        end_time = datetime.now()
        duration = end_time - start_time

        print("\n" + "=" * 70)
        print("PIPELINE COMPLETE")
        print("=" * 70)
        print(f"Duration: {duration}")
        print(f"Status: {'SUCCESS' if all_success else 'COMPLETED WITH ERRORS'}")
        print(f"Output: {self.config.output_dir}")

        # Print final progress
        self.state.print_progress()

        return all_success

    async def run_stage(self, stage_name: str) -> bool:
        """Run a single stage.

        Args:
            stage_name: Name of stage to run

        Returns:
            True if stage completed successfully

        Raises:
            ValueError: If stage name not found
        """
        stage_class = None
        for name, cls in STAGE_ORDER:
            if name == stage_name:
                stage_class = cls
                break

        if stage_class is None:
            raise ValueError(f"Unknown stage: {stage_name}")

        stage = stage_class(self.config, self.state)

        if stage_name == "activation_logging":
            return await stage.run()
        elif self.config.use_batch_api:
            return await stage.run_batch()
        else:
            return await stage.run()

    def reset_stage(self, stage_name: str) -> None:
        """Reset progress for a specific stage.

        Args:
            stage_name: Name of stage to reset
        """
        self.state.reset_stage(stage_name)
        print(f"Reset progress for stage: {stage_name}")

    def reset_all(self) -> None:
        """Reset all pipeline progress."""
        self.state.reset_all()
        print("Reset all pipeline progress")

    def get_output_summary(self) -> Dict:
        """Get summary of pipeline outputs.

        Returns:
            Dict with output file info
        """
        outputs = {}

        output_files = [
            "cards.jsonl",
            "scenarios.jsonl",
            "paraphrases.jsonl",
            "activations.h5",
            "activation_metadata.json",
            "state.json",
            "run_manifest.json",
        ]

        for filename in output_files:
            path = self.config.output_dir / filename
            if path.exists():
                outputs[filename] = {
                    "exists": True,
                    "size_bytes": path.stat().st_size,
                    "modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
                }
            else:
                outputs[filename] = {"exists": False}

        return outputs

    def print_output_summary(self) -> None:
        """Print summary of pipeline outputs."""
        summary = self.get_output_summary()

        print("\n" + "=" * 60)
        print("OUTPUT FILES")
        print("=" * 60)

        for filename, info in summary.items():
            if info["exists"]:
                size_kb = info["size_bytes"] / 1024
                print(f"  {filename}: {size_kb:.1f} KB ({info['modified']})")
            else:
                print(f"  {filename}: (not created)")

        print("=" * 60 + "\n")


async def run_pipeline(config_path: Path) -> bool:
    """Convenience function to run pipeline from config file.

    Args:
        config_path: Path to YAML config file

    Returns:
        True if pipeline completed successfully
    """
    config = AppraisalConfig.from_yaml(config_path)
    config.print_summary()

    pipeline = AppraisalPipeline(config)
    return await pipeline.run()
