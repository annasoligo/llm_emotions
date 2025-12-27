"""cPCA experiment runner - minimal implementation."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .base import ExperimentConfig, BaseExperiment
from ..core import load_activations_hdf5, save_cpca_results
from ..methods import run_cpca_all_layers


@dataclass
class CPCAConfig(ExperimentConfig):
    """Configuration for cPCA experiments."""

    # Data
    data_path: Path = None
    model_name: str = "unknown"

    # cPCA parameters
    alpha: Optional[float] = None  # None = auto-tune
    alpha_range: tuple[float, float] = (0.1, 1000.0)
    n_alphas: int = 40
    n_components: int = 50
    use_diffs: bool = True

    # Optional tier filtering
    tier: Optional[str] = None

    def __post_init__(self):
        """Validate cPCA-specific config."""
        # Call parent validation
        ExperimentConfig.__post_init__(self)

        if self.data_path is None:
            raise ValueError("data_path is required")

        # Convert to Path
        if isinstance(self.data_path, str):
            self.data_path = Path(self.data_path)

        if not isinstance(self.data_path, Path):
            raise ValueError(f"data_path must be Path or str, got {type(self.data_path)}")

        if self.alpha is not None and self.alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {self.alpha}")

        if self.n_components <= 0:
            raise ValueError(f"n_components must be positive, got {self.n_components}")

        if self.n_alphas <= 0:
            raise ValueError(f"n_alphas must be positive, got {self.n_alphas}")

        if len(self.alpha_range) != 2:
            raise ValueError(f"alpha_range must have 2 elements, got {len(self.alpha_range)}")

        if self.alpha_range[0] >= self.alpha_range[1]:
            raise ValueError(
                f"alpha_range[0] must be < alpha_range[1], "
                f"got {self.alpha_range}"
            )


class CPCAExperiment(BaseExperiment):
    """cPCA experiment runner."""

    def __init__(self, config: CPCAConfig):
        """Initialize cPCA experiment.

        Args:
            config: cPCA configuration

        Raises:
            ValueError: If config invalid
        """
        if not isinstance(config, CPCAConfig):
            raise ValueError(
                f"config must be CPCAConfig instance, got {type(config)}"
            )

        super().__init__(config)
        self.config: CPCAConfig = config

    def load_data(self) -> None:
        """Load activation data.

        Raises:
            FileNotFoundError: If data file not found
            KeyError: If data format invalid
            ValueError: If data contains invalid values
        """
        print(f"Loading activations from: {self.config.data_path}")

        self.activations, self.metadata, self.attrs = load_activations_hdf5(
            self.config.data_path
        )

        print(f"Loaded {len(self.activations)} activation pairs")

        # Optional: Filter by tier
        if self.config.tier is not None:
            print(f"Filtering to tier: {self.config.tier}")

            id_to_meta = {m["id"]: m for m in self.metadata}
            tier_ids = {
                pid
                for pid in self.activations.keys()
                if id_to_meta.get(pid, {}).get("tier") == self.config.tier
            }

            if not tier_ids:
                raise ValueError(
                    f"No activations found for tier '{self.config.tier}'"
                )

            self.activations = {
                pid: acts for pid, acts in self.activations.items() if pid in tier_ids
            }
            self.metadata = [m for m in self.metadata if m["id"] in tier_ids]

            print(f"Filtered to {len(self.activations)} pairs")

    def run(self) -> None:
        """Run cPCA on all layers.

        Raises:
            ValueError: If execution fails
        """
        print("Running cPCA...")

        if self.config.alpha is None:
            print(
                f"Auto-tuning alpha per layer (range: {self.config.alpha_range}, "
                f"n_alphas: {self.config.n_alphas})"
            )
        else:
            print(f"Using fixed alpha: {self.config.alpha}")

        self.results = run_cpca_all_layers(
            activations=self.activations,
            metadata=self.metadata,
            alpha=self.config.alpha,
            n_components=self.config.n_components,
            alpha_range=self.config.alpha_range,
            n_alphas=self.config.n_alphas,
            use_diffs=self.config.use_diffs,
        )

        print(f"cPCA complete: {self.results['config']['num_layers']} layers processed")

    def save_results(self) -> None:
        """Save cPCA results.

        Raises:
            OSError: If cannot write files
        """
        # Prepare results for saving
        num_layers = self.results["config"]["num_layers"]
        n_components = self.results["config"]["n_components"]

        # Get hidden dim from first layer
        hidden_dim = self.results["components"][0].shape[1]

        # Stack components and eigenvalues into arrays
        components_array = []
        eigenvalues_array = []
        alphas_array = []

        for layer_idx in range(num_layers):
            components_array.append(self.results["components"][layer_idx])
            eigenvalues_array.append(self.results["eigenvalues"][layer_idx])
            alphas_array.append(self.results["alpha_per_layer"][layer_idx])

        import numpy as np

        save_results = {
            "components": np.stack(components_array),  # [num_layers, n_components, hidden_dim]
            "eigenvalues": np.stack(eigenvalues_array),  # [num_layers, n_components]
            "alphas": np.array(alphas_array),  # [num_layers]
            "config": self.results["config"],
            "metadata": self.results["metadata"],
            "pair_ids": self.results["pair_ids"],
        }

        # Save main results
        output_file = self.config.output_dir / f"{self.config.model_name}_cpca.npz"
        save_cpca_results(save_results, output_file)

        print(f"Saved results to: {output_file}")

        # Save summary
        from ..core.saving import save_json

        summary = {
            "model_name": self.config.model_name,
            "n_components": n_components,
            "num_layers": num_layers,
            "num_pairs": len(self.results["pair_ids"]),
            "hidden_dim": hidden_dim,
            "use_diffs": self.config.use_diffs,
            "alpha": self.config.alpha,
        }

        summary_file = self.config.output_dir / f"{self.config.model_name}_cpca_summary.json"
        save_json(summary, summary_file)

        print(f"Saved summary to: {summary_file}")
