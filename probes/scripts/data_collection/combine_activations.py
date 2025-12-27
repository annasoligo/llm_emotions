#!/usr/bin/env python3
"""Combine emotional and neutral conversation activations.

This unified script supports two modes:
1. Standard mode: Creates single HDF5 file with nested structure (global + regional + special)
2. Regional mode: Creates separate HDF5 files per region for independent cPCA

Usage:
    # Standard mode (all activation types in one file)
    python -m probes.scripts.combine_activations \\
        --type standard \\
        --emotional data/activations/conversations2.h5 \\
        --neutral data/activations/conversations2_neutral.h5 \\
        --output data/activations/conversations2_combined.h5

    # Regional mode (separate files per region)
    python -m probes.scripts.combine_activations \\
        --type regional \\
        --emotional data/activations/conversations2.h5 \\
        --neutral data/activations/conversations2_neutral.h5 \\
        --output-dir data/activations/regional
"""

import argparse
import json
from pathlib import Path
from typing import List

import h5py
from tqdm import tqdm


def load_and_validate_files(emotional_path: Path, neutral_path: Path):
    """Load and validate emotional and neutral activation files.

    Args:
        emotional_path: Path to emotional activations HDF5
        neutral_path: Path to neutral activations HDF5

    Returns:
        (emotional_h5, neutral_h5, conv_ids, metadata)

    Raises:
        ValueError: If files don't match
    """
    print(f"Loading emotional: {emotional_path}")
    print(f"Loading neutral: {neutral_path}")

    f_emo = h5py.File(emotional_path, 'r')
    f_neu = h5py.File(neutral_path, 'r')

    # Get metadata
    meta_emo = json.loads(f_emo['metadata'][()])
    meta_neu = json.loads(f_neu['metadata'][()])

    print(f"Emotional conversations: {len(meta_emo)}")
    print(f"Neutral conversations: {len(meta_neu)}")

    if len(meta_emo) != len(meta_neu):
        raise ValueError(f"Mismatch: {len(meta_emo)} emotional vs {len(meta_neu)} neutral")

    # Verify IDs match
    emo_ids = sorted([m['id'] for m in meta_emo])
    neu_ids = sorted([m['id'] for m in meta_neu])

    if emo_ids != neu_ids:
        raise ValueError("Conversation IDs don't match between files!")

    return f_emo, f_neu, emo_ids, meta_emo


def combine_standard(
    f_emo: h5py.File,
    f_neu: h5py.File,
    conv_ids: List[str],
    metadata: List[dict],
    output_path: Path,
):
    """Combine activations in standard mode (single nested file).

    Args:
        f_emo: Emotional activations HDF5 file (open)
        f_neu: Neutral activations HDF5 file (open)
        conv_ids: List of conversation IDs
        metadata: Metadata list
        output_path: Output HDF5 path
    """
    print(f"\nCreating combined file: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, 'w') as f_out:
        # Save metadata (use emotional metadata with emotions preserved)
        f_out.create_dataset('metadata', data=json.dumps(metadata))

        # Create activations group
        acts_group = f_out.create_group('activations')

        # For each conversation, save as pair
        for conv_id in tqdm(conv_ids, desc="Combining conversations"):
            # Create group for this pair
            pair_group = acts_group.create_group(conv_id)

            # Copy emotional and neutral activations
            emo_conv = f_emo['activations'][conv_id]
            neu_conv = f_neu['activations'][conv_id]

            # Global activations
            emo_global = emo_conv['global'][:]
            neu_global = neu_conv['global'][:]

            pair_group.create_dataset('emotional', data=emo_global, compression='gzip')
            pair_group.create_dataset('neutral', data=neu_global, compression='gzip')

            # Regional activations (if they exist)
            if 'regional' in emo_conv:
                regional_group = pair_group.create_group('regional')

                for region_name in emo_conv['regional'].keys():
                    emo_regional = emo_conv['regional'][region_name][:]
                    neu_regional = neu_conv['regional'][region_name][:]

                    region_pair = regional_group.create_group(region_name)
                    region_pair.create_dataset('emotional', data=emo_regional, compression='gzip')
                    region_pair.create_dataset('neutral', data=neu_regional, compression='gzip')

            # Special token activations (if they exist)
            if 'special' in emo_conv:
                special_group = pair_group.create_group('special')

                for special_name in emo_conv['special'].keys():
                    emo_special = emo_conv['special'][special_name][:]
                    neu_special = neu_conv['special'][special_name][:]

                    special_pair = special_group.create_group(special_name)
                    special_pair.create_dataset('emotional', data=emo_special, compression='gzip')
                    special_pair.create_dataset('neutral', data=neu_special, compression='gzip')

    print(f"\nDone! Combined {len(conv_ids)} conversation pairs")
    print(f"Saved to: {output_path}")


def combine_regional(
    f_emo: h5py.File,
    f_neu: h5py.File,
    conv_ids: List[str],
    metadata: List[dict],
    output_dir: Path,
):
    """Combine activations in regional mode (separate files per region).

    Args:
        f_emo: Emotional activations HDF5 file (open)
        f_neu: Neutral activations HDF5 file (open)
        conv_ids: List of conversation IDs
        metadata: Metadata list
        output_dir: Output directory for regional files
    """
    # Get available regions from first conversation
    first_conv_id = conv_ids[0]
    emo_conv = f_emo[f"activations/{first_conv_id}"]

    if "regional" not in emo_conv:
        raise ValueError("No regional activations found in emotional file")

    region_names = list(emo_conv["regional"].keys())
    print(f"\nRegions found: {region_names}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create combined file for each region
    for region_name in region_names:
        output_path = output_dir / f"conversations2_combined_{region_name}.h5"
        print(f"\nCreating combined file for region '{region_name}': {output_path}")

        with h5py.File(output_path, "w") as f_out:
            # Copy attributes
            for key, value in f_emo.attrs.items():
                f_out.attrs[key] = value

            # Copy metadata
            f_out.create_dataset("metadata", data=json.dumps(metadata))

            # Create activations group
            acts_group = f_out.create_group("activations")

            # Combine conversations for this region
            for conv_id in tqdm(conv_ids, desc=f"Combining {region_name}"):
                emo_conv = f_emo[f"activations/{conv_id}"]
                neu_conv = f_neu[f"activations/{conv_id}"]

                # Get regional activations
                emo_regional = emo_conv[f"regional/{region_name}"][:]
                neu_regional = neu_conv[f"regional/{region_name}"][:]

                # Create pair group
                pair_group = acts_group.create_group(conv_id)
                pair_group.create_dataset(
                    "emotional", data=emo_regional, compression="gzip"
                )
                pair_group.create_dataset(
                    "neutral", data=neu_regional, compression="gzip"
                )

        print(f"✓ Created {output_path}")

    print(f"\n✓ All regional combined files created in {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Combine emotional and neutral conversation activations"
    )
    parser.add_argument(
        "--type",
        type=str,
        required=True,
        choices=["standard", "regional"],
        help="Combination mode: 'standard' (single nested file) or 'regional' (separate files per region)",
    )
    parser.add_argument(
        "--emotional",
        type=Path,
        required=True,
        help="Path to emotional activations HDF5",
    )
    parser.add_argument(
        "--neutral",
        type=Path,
        required=True,
        help="Path to neutral activations HDF5",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output HDF5 file (required for --type standard)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (required for --type regional)",
    )

    args = parser.parse_args()

    # Validate arguments based on type
    if args.type == "standard" and not args.output:
        parser.error("--output is required for --type standard")
    if args.type == "regional" and not args.output_dir:
        parser.error("--output-dir is required for --type regional")

    # Load and validate files
    f_emo, f_neu, conv_ids, metadata = load_and_validate_files(
        args.emotional, args.neutral
    )

    try:
        # Run appropriate combination mode
        if args.type == "standard":
            combine_standard(f_emo, f_neu, conv_ids, metadata, args.output)
        else:  # regional
            combine_regional(f_emo, f_neu, conv_ids, metadata, args.output_dir)
    finally:
        # Clean up file handles
        f_emo.close()
        f_neu.close()


if __name__ == "__main__":
    main()
