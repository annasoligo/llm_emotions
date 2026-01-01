"""Load baseline statistics for steering magnitude calculation."""

import json
from pathlib import Path
from typing import Dict


class BaselineMagnitudeCalculator:
    """Calculate steering magnitudes from baseline statistics."""

    def __init__(self, baseline_dir: Path, aggregation: str = 'first_assistant_token'):
        self.baseline_dir = Path(baseline_dir)
        self.aggregation = aggregation
        self.layer_stats: Dict[int, Dict[str, float]] = {}
        self._load_stats()

    def _load_stats(self):
        """Load statistics from JSON files."""
        for stats_file in self.baseline_dir.glob('layer*_stats.json'):
            layer_num = int(stats_file.stem.replace('layer', '').replace('_stats', ''))

            with open(stats_file) as f:
                data = json.load(f)

            agg_data = data['aggregations'][self.aggregation]
            self.layer_stats[layer_num] = {
                'mean': agg_data['mean'],
                'std': agg_data['std']
            }

    def get_steering_magnitude(self, layer: int, n_std: float = 1.0) -> float:
        """Get steering magnitude as multiple of standard deviation."""
        return self.layer_stats[layer]['std'] * n_std

    def get_cap_threshold(self, layer: int, n_std_above_mean: float = 0.5) -> float:
        """Get capping threshold as mean + N std."""
        stats = self.layer_stats[layer]
        return stats['mean'] + n_std_above_mean * stats['std']

    def get_stats(self, layer: int) -> Dict[str, float]:
        """Get raw statistics for layer."""
        return self.layer_stats[layer].copy()
