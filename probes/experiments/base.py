"""Base experiment classes - minimal, no fallbacks."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any
import yaml


@dataclass
class ExperimentConfig:
    """Base configuration for all experiments."""

    name: str
    output_dir: Path
    seed: int = 42
    device: str = "cuda"

    def __post_init__(self):
        """Validate config after initialization."""
        if not self.name:
            raise ValueError("name cannot be empty")

        # Convert output_dir to Path if string
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)

        if not isinstance(self.output_dir, Path):
            raise ValueError(f"output_dir must be Path or str, got {type(self.output_dir)}")

        if self.seed < 0:
            raise ValueError(f"seed must be non-negative, got {self.seed}")

        if self.device not in ("cuda", "cpu", "mps"):
            raise ValueError(f"device must be 'cuda', 'cpu', or 'mps', got {self.device}")

    @classmethod
    def from_yaml(cls, path: Path):
        """Load config from YAML file.

        Args:
            path: Path to YAML file

        Returns:
            Config instance

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If YAML invalid or missing required fields
        """
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            try:
                config_dict = yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise ValueError(f"Invalid YAML in {path}: {e}")

        if not isinstance(config_dict, dict):
            raise ValueError(f"Expected dict from YAML, got {type(config_dict)}")

        try:
            return cls(**config_dict)
        except TypeError as e:
            raise ValueError(f"Invalid config fields in {path}: {e}")

    def to_yaml(self, path: Path) -> None:
        """Save config to YAML file.

        Args:
            path: Path to save YAML file

        Raises:
            OSError: If cannot write file
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = asdict(self)

        # Convert Path to str for YAML serialization
        for key, value in config_dict.items():
            if isinstance(value, Path):
                config_dict[key] = str(value)

        with open(path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)


class BaseExperiment(ABC):
    """Base class for all experiments - fail fast, no fallbacks."""

    def __init__(self, config: ExperimentConfig):
        """Initialize experiment.

        Args:
            config: Experiment configuration

        Raises:
            ValueError: If config invalid
        """
        if not isinstance(config, ExperimentConfig):
            raise ValueError(
                f"config must be ExperimentConfig instance, got {type(config)}"
            )

        self.config = config
        self._setup_output_dir()
        self._set_random_seed()

    def _setup_output_dir(self) -> None:
        """Create output directory structure.

        Raises:
            OSError: If cannot create directories
        """
        try:
            self.config.output_dir.mkdir(parents=True, exist_ok=True)
            (self.config.output_dir / "checkpoints").mkdir(exist_ok=True)
            (self.config.output_dir / "logs").mkdir(exist_ok=True)
        except OSError as e:
            raise OSError(f"Failed to create output directories: {e}")

    def _set_random_seed(self) -> None:
        """Set random seed for reproducibility."""
        import random
        import numpy as np

        random.seed(self.config.seed)
        np.random.seed(self.config.seed)

        # Set torch seed if available
        try:
            import torch

            torch.manual_seed(self.config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.config.seed)
        except ImportError:
            pass  # Torch not installed, skip

    @abstractmethod
    def load_data(self) -> None:
        """Load experiment data.

        Must raise informative errors if data cannot be loaded.
        """
        pass

    @abstractmethod
    def run(self) -> None:
        """Run the experiment.

        Must raise informative errors if execution fails.
        """
        pass

    @abstractmethod
    def save_results(self) -> None:
        """Save experiment results.

        Must raise informative errors if saving fails.
        """
        pass

    def execute(self) -> None:
        """Full experiment pipeline.

        Raises:
            Exception: Any exception from load_data, run, or save_results
        """
        print(f"Starting experiment: {self.config.name}")
        print(f"Output directory: {self.config.output_dir}")

        # Save config
        config_path = self.config.output_dir / "config.yaml"
        self.config.to_yaml(config_path)
        print(f"Saved config to: {config_path}")

        self.load_data()
        self.run()
        self.save_results()

        print(f"Experiment complete: {self.config.name}")
