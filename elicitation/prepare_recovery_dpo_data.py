"""
Prepare combined DPO data with baseline calm pairs AND recovery pairs.

Structure:
- Pair type 1 (baseline): Calm vs Frustrated (existing data)
- Pairs type 2-4 (recovery): Frustrated-prefix + recovery vs Frustrated-full
"""

import json
import random
import argparse
from pathlib import Path
from collections import defaultdict


def load_jsonl(filepath: str) -> list:
    """Load JSONL file."""
    data = []
    with open(filepath) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def main():
    parser = argparse.ArgumentParser(description="Combine baseline and recovery DPO data")
    parser.add_argument("--baseline-file", type=str, required=True,
                        help="File with baseline calm vs frustrated pairs")
    parser.add_argument("--recovery-file", type=str, required=True,
                        help="File with recovery pairs")
    parser.add_argument("--output-file", type=str, required=True,
                        help="Output combined DPO data")
    parser.add_argument("--baseline-weight", type=float, default=1.0,
                        help="Weight for baseline pairs (1.0 = keep all)")
    parser.add_argument("--recovery-weight", type=float, default=1.0,
                        help="Weight for recovery pairs (1.0 = keep all)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()
    random.seed(args.seed)

    print("=" * 60)
    print("COMBINING BASELINE + RECOVERY DPO DATA")
    print("=" * 60)

    # Load baseline pairs
    print(f"\nLoading baseline pairs from {args.baseline_file}...")
    baseline_pairs = load_jsonl(args.baseline_file)
    print(f"  Loaded {len(baseline_pairs)} baseline pairs")

    # Load recovery pairs
    print(f"\nLoading recovery pairs from {args.recovery_file}...")
    recovery_pairs = load_jsonl(args.recovery_file)
    print(f"  Loaded {len(recovery_pairs)} recovery pairs")

    # Apply weights (subsample if weight < 1.0)
    if args.baseline_weight < 1.0:
        n_keep = int(len(baseline_pairs) * args.baseline_weight)
        baseline_pairs = random.sample(baseline_pairs, n_keep)
        print(f"  Subsampled to {len(baseline_pairs)} baseline pairs")

    if args.recovery_weight < 1.0:
        n_keep = int(len(recovery_pairs) * args.recovery_weight)
        recovery_pairs = random.sample(recovery_pairs, n_keep)
        print(f"  Subsampled to {len(recovery_pairs)} recovery pairs")

    # Mark pair types
    for pair in baseline_pairs:
        pair['pair_type'] = 'baseline'

    for pair in recovery_pairs:
        pair['pair_type'] = f"recovery_{pair.get('recovery_point', 'unknown')}"

    # Combine and shuffle
    combined = baseline_pairs + recovery_pairs
    random.shuffle(combined)

    # Save
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for pair in combined:
            f.write(json.dumps(pair) + '\n')

    # Statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total pairs: {len(combined)}")

    # By pair type
    by_type = defaultdict(int)
    for pair in combined:
        by_type[pair['pair_type']] += 1

    print("\nBy pair type:")
    for ptype in sorted(by_type.keys()):
        pct = 100 * by_type[ptype] / len(combined)
        print(f"  {ptype}: {by_type[ptype]} ({pct:.1f}%)")

    # By turn
    by_turn = defaultdict(int)
    for pair in combined:
        by_turn[pair.get('turn', 1)] += 1

    print("\nBy turn:")
    for turn in sorted(by_turn.keys()):
        print(f"  Turn {turn}: {by_turn[turn]}")

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
