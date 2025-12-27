#!/usr/bin/env python3
"""Extract regional activation views from combined conversation file.

The combined file has structure:
  activations/{id}/emotional (global)
  activations/{id}/neutral (global)
  activations/{id}/regional/{region}/emotional
  activations/{id}/regional/{region}/neutral

This script extracts regional pairs into flat structure for cPCA.
"""

import argparse
import json
from pathlib import Path

import h5py
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--regions", nargs="+", default=["user", "asst", "special1", "special2"])
    args = parser.parse_args()

    print(f"Input: {args.input}")
    print(f"Output dir: {args.output_dir}")
    print(f"Regions: {args.regions}")
    print()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.input, 'r') as f_in:
        # Load metadata
        metadata = json.loads(f_in['metadata'][()])
        conv_ids = list(f_in['activations'].keys())

        print(f"Found {len(conv_ids)} conversations")

        # Extract each region
        for region in args.regions:
            output_path = args.output_dir / f"conversations2_regional_{region}.h5"
            print(f"\nExtracting {region} → {output_path}")

            with h5py.File(output_path, 'w') as f_out:
                acts_group = f_out.create_group('activations')

                for conv_id in tqdm(conv_ids, desc=f"  {region}"):
                    conv_group = f_in[f'activations/{conv_id}']

                    # Check regional data exists
                    if 'regional' not in conv_group:
                        continue
                    if region not in conv_group['regional']:
                        continue

                    region_group = conv_group[f'regional/{region}']

                    if 'emotional' not in region_group or 'neutral' not in region_group:
                        continue

                    # Extract
                    emo_acts = region_group['emotional'][:]
                    neu_acts = region_group['neutral'][:]

                    # Save pair
                    pair_group = acts_group.create_group(conv_id)
                    pair_group.create_dataset('emotional', data=emo_acts, compression='gzip')
                    pair_group.create_dataset('neutral', data=neu_acts, compression='gzip')

                # Copy metadata and attributes
                f_out.create_dataset('metadata', data=f_in['metadata'][()])
                for key, value in f_in.attrs.items():
                    f_out.attrs[key] = value
                f_out.attrs['region'] = region
                num_pairs = len(acts_group)
                f_out.attrs['num_pairs'] = num_pairs

            print(f"  ✓ Extracted {num_pairs} pairs")

    print("\n✓ All regions extracted")


if __name__ == "__main__":
    main()
