#!/usr/bin/env python3
"""Run cPCA experiment from config file.

Usage:
    python scripts/run_cpca.py experiments/configs/cpca_example.yaml
"""

import argparse
import sys
from pathlib import Path

from probes.experiments.cpca_experiment import CPCAExperiment, CPCAConfig


def main():
    parser = argparse.ArgumentParser(description="Run cPCA experiment")
    parser.add_argument(
        "config",
        type=Path,
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    # Load config
    try:
        config = CPCAConfig.from_yaml(args.config)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

    # Run experiment
    try:
        experiment = CPCAExperiment(config)
        experiment.execute()
    except Exception as e:
        print(f"Error running experiment: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
