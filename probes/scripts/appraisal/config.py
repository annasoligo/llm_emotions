"""Pydantic configuration schema for appraisal pipeline."""

from pathlib import Path
from typing import List, Optional, Literal
import yaml

from pydantic import BaseModel, Field, field_validator


class AxisConfig(BaseModel):
    """Configuration for a single appraisal axis."""

    name: str = Field(..., description="Short name for the axis (e.g., 'agency')")
    description: str = Field(..., description="Full description of the axis hypothesis")


class DataGenerationConfig(BaseModel):
    """Configuration for data generation stages."""

    axes: List[AxisConfig] = Field(..., description="List of axis hypotheses to process")
    domains: List[str] = Field(
        default_factory=lambda: [
            "technical_debugging",
            "medical_triage",
            "financial_planning",
            "project_management",
            "customer_support",
        ],
        description="Domains for scenario generation",
    )
    n_cards_per_axis: int = Field(default=5, ge=1, le=20, description="Number of cards per axis")
    n_scenarios_per_card: int = Field(default=3, ge=1, le=10, description="Number of scenarios per card")
    n_paraphrases: int = Field(default=3, ge=1, le=10, description="Number of paraphrases per scenario")


class ActivationConfig(BaseModel):
    """Configuration for activation logging stage."""

    target_model: str = Field(
        default="google/gemma-2-9b-it",
        description="HuggingFace model ID for activation logging",
    )
    layers: Optional[List[int]] = Field(
        default=None,
        description="Specific layers to log (None = all layers)",
    )
    representations: List[str] = Field(
        default_factory=lambda: ["assistant_start_last_token", "boundary_special_tokens_mean"],
        description="Types of representations to extract",
    )
    batch_size: int = Field(default=1, ge=1, le=32, description="Batch size for activation logging")
    dtype: Literal["float32", "float16", "bfloat16"] = Field(
        default="bfloat16",
        description="Data type for model loading",
    )

    @field_validator("representations")
    @classmethod
    def validate_representations(cls, v):
        valid = {"assistant_start_last_token", "boundary_special_tokens_mean"}
        for rep in v:
            if rep not in valid:
                raise ValueError(f"Invalid representation '{rep}'. Must be one of: {valid}")
        return v


class AppraisalConfig(BaseModel):
    """Main configuration for appraisal pipeline."""

    name: str = Field(..., description="Run name for this pipeline execution")
    output_dir: Path = Field(..., description="Directory for all outputs")

    # LLM settings
    claude_model: str = Field(
        default="claude-opus-4-5-20251101",
        description="Claude model for data generation",
    )
    use_batch_api: bool = Field(
        default=True,
        description="Use Anthropic Batch API (recommended for cost/rate limits)",
    )
    max_tokens: int = Field(default=4096, ge=256, le=16384, description="Max tokens per LLM response")
    concurrency: int = Field(default=1, ge=1, le=100, description="Number of concurrent API requests")

    # Data generation settings
    data_generation: DataGenerationConfig = Field(
        default_factory=DataGenerationConfig,
        description="Data generation configuration",
    )

    # Activation settings
    activation: ActivationConfig = Field(
        default_factory=ActivationConfig,
        description="Activation logging configuration",
    )

    # Validation settings
    additional_forbidden_words: List[str] = Field(
        default_factory=list,
        description="Additional words to forbid in scenarios",
    )
    max_diff_ratio: float = Field(
        default=0.20,
        ge=0.05,
        le=0.50,
        description="Maximum difference ratio for minimal pairs",
    )

    # Resume settings
    resume: bool = Field(
        default=True,
        description="Resume from last checkpoint if available",
    )

    # Stage control
    stages_to_run: Optional[List[str]] = Field(
        default=None,
        description="Specific stages to run (None = all stages)",
    )
    skip_activation_logging: bool = Field(
        default=False,
        description="Skip activation logging stage (useful for data-only runs)",
    )

    @field_validator("output_dir", mode="before")
    @classmethod
    def expand_path(cls, v):
        return Path(v).expanduser().resolve()

    @classmethod
    def from_yaml(cls, path: Path) -> "AppraisalConfig":
        """Load configuration from YAML file.

        Args:
            path: Path to YAML config file

        Returns:
            AppraisalConfig instance

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config is invalid
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        return cls(**data)

    def to_yaml(self, path: Path) -> None:
        """Save configuration to YAML file.

        Args:
            path: Path for output YAML file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict with custom serialization
        data = self.model_dump(mode='json')

        # Convert Path to string
        data['output_dir'] = str(self.output_dir)

        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def to_dict(self) -> dict:
        """Convert config to dict for state management."""
        return self.model_dump(mode='json')

    def get_output_path(self, filename: str) -> Path:
        """Get full path for an output file.

        Args:
            filename: Relative filename

        Returns:
            Full path in output directory
        """
        return self.output_dir / filename

    def print_summary(self) -> None:
        """Print configuration summary to console."""
        print("\n" + "=" * 60)
        print("APPRAISAL PIPELINE CONFIGURATION")
        print("=" * 60)
        print(f"Name: {self.name}")
        print(f"Output: {self.output_dir}")
        print(f"Claude model: {self.claude_model}")
        print(f"Batch API: {self.use_batch_api}")
        print(f"Concurrency: {self.concurrency}")
        print()
        print("Data Generation:")
        print(f"  Axes: {len(self.data_generation.axes)}")
        print(f"  Domains: {len(self.data_generation.domains)}")
        print(f"  Cards per axis: {self.data_generation.n_cards_per_axis}")
        print(f"  Scenarios per card: {self.data_generation.n_scenarios_per_card}")
        print(f"  Paraphrases: {self.data_generation.n_paraphrases}")
        print()
        print("Activation Logging:")
        print(f"  Target model: {self.activation.target_model}")
        print(f"  Layers: {self.activation.layers or 'all'}")
        print(f"  Representations: {self.activation.representations}")
        print()
        print(f"Resume: {self.resume}")
        print(f"Skip activation logging: {self.skip_activation_logging}")
        print("=" * 60 + "\n")


def create_test_config(output_dir: Path) -> AppraisalConfig:
    """Create a minimal test configuration.

    Args:
        output_dir: Output directory for test

    Returns:
        AppraisalConfig with minimal settings for testing
    """
    return AppraisalConfig(
        name="test_run",
        output_dir=output_dir,
        data_generation=DataGenerationConfig(
            axes=[
                AxisConfig(
                    name="agency",
                    description="Whether the actor has control over outcomes vs being at the mercy of external factors",
                ),
            ],
            domains=["technical_debugging"],
            n_cards_per_axis=1,
            n_scenarios_per_card=1,
            n_paraphrases=1,
        ),
        skip_activation_logging=True,
    )
