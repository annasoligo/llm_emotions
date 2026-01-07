#!/usr/bin/env python3
"""
Complete preprocessing wrapper for eval dashboard.
Orchestrates all preprocessing steps in optimal order.
"""
import sys
import argparse
from pathlib import Path
import subprocess

# Add paths
sys.path.insert(0, '/workspace-vast/annas/git/research-tools')
sys.path.insert(0, "/workspace-vast/annas/git/believe-it-or-not")

# Import preprocessing modules
from eval_dashboard import data_preprocessing
from eval_dashboard import add_logit_lens_to_existing
from eval_dashboard import add_axis_lens_to_existing

# Default probe list (all available probes)
DEFAULT_PROBES = [
    'orthogonal_raw',
    'orthogonal_cpca_top10',
    'orthogonal_regularized_lambda100',
    'text_raw',
    'text_cpca',
    'centroid_k10',
    'diverse_isolation_lambda10'
]

def print_header(title, step=None):
    """Print a formatted header."""
    print("\n" + "="*80)
    if step:
        print(f"STEP {step}: {title}")
    else:
        print(title)
    print("="*80 + "\n")


def run_probe_preprocessing(input_path, output_path, probe_keys):
    """Run standard probe-based preprocessing."""
    print_header("PROBE-BASED PREPROCESSING", "1/3")

    print("Running data_preprocessing.py with probe types:")
    for probe in probe_keys:
        print(f"  - {probe}")
    print()

    try:
        data_preprocessing.preprocess_all_conversations(
            data_path=str(input_path),
            probe_keys=probe_keys,
            output_path=str(output_path),
            use_simple_splitter=False
        )
        print("\n✓ Probe-based preprocessing complete")
        return True
    except Exception as e:
        print(f"\n✗ Probe-based preprocessing failed: {e}")
        return False


def run_logit_lens_preprocessing(pickle_path):
    """Run logit lens preprocessing with all layer ranges."""
    print_header("LOGIT LENS PREPROCESSING", "2/3")

    print("Running add_logit_lens_to_existing.py with:")
    print("  - All layer ranges (L40-50, L30-40, L20-30)")
    print("  - Integrated baseline correction")
    print("  - Per-layer scores for layerwise plotting")
    print()

    # Import and modify the module's globals
    add_logit_lens_to_existing.input_path = pickle_path
    add_logit_lens_to_existing.output_path = pickle_path

    try:
        add_logit_lens_to_existing.main()
        print("\n✓ Logit lens preprocessing complete")
        return True
    except Exception as e:
        print(f"\n✗ Logit lens preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_axis_lens_preprocessing(pickle_path):
    """Run axis lens preprocessing."""
    print_header("AXIS LENS PREPROCESSING", "3/3")

    print("Running add_axis_lens_to_existing.py with:")
    print("  - Integrated baseline correction")
    print("  - Random token baseline computation")
    print()

    # Modify sys.argv for the script
    old_argv = sys.argv
    sys.argv = ['add_axis_lens_to_existing.py', str(pickle_path), str(pickle_path)]

    try:
        add_axis_lens_to_existing.main()
        print("\n✓ Axis lens preprocessing complete")
        return True
    except Exception as e:
        print(f"\n✗ Axis lens preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        sys.argv = old_argv


def main():
    parser = argparse.ArgumentParser(
        description='Complete preprocessing for eval dashboard (probes + logit lens + axis lens)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (process default input)
  python preprocess_complete.py

  # Custom input/output
  python preprocess_complete.py --input data.jsonl --output data.pkl

  # Subset of probes only
  python preprocess_complete.py --probes orthogonal_raw text_raw logit_lens_mean

  # Skip certain steps
  python preprocess_complete.py --skip-probes  # Start from existing pickle
  python preprocess_complete.py --skip-logit   # Skip logit lens
  python preprocess_complete.py --skip-axis    # Skip axis lens

Probe types available:
  Standard probes:
    - orthogonal_raw, orthogonal_cpca_top10, orthogonal_regularized_lambda100
    - text_raw, text_cpca
    - centroid_k10
    - diverse_isolation_lambda10

  Logit lens (automatically added):
    - logit_lens_mean (L40-50), logit_lens_mean_l30_40, logit_lens_mean_l20_30

  Axis lens (automatically added):
    - axis_lens_mean
        """
    )

    parser.add_argument(
        '--input', '-i',
        type=str,
        default='/workspace-vast/annas/git/research-tools/elicitation/outputs/annotated_emotion_onset_gemma3.jsonl',
        help='Path to emotion onset data (JSONL)'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default='/workspace-vast/annas/git/research-tools/eval_dashboard/data/preprocessed_conversations_complete.pkl',
        help='Output path for preprocessed pickle'
    )

    parser.add_argument(
        '--probes', '-p',
        type=str,
        nargs='+',
        default=DEFAULT_PROBES,
        help='Probe types to include (default: all)'
    )

    parser.add_argument(
        '--skip-probes',
        action='store_true',
        help='Skip probe preprocessing (start from existing pickle)'
    )

    parser.add_argument(
        '--skip-logit',
        action='store_true',
        help='Skip logit lens preprocessing'
    )

    parser.add_argument(
        '--skip-axis',
        action='store_true',
        help='Skip axis lens preprocessing'
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    # Print configuration
    print("="*80)
    print("COMPLETE PREPROCESSING FOR EVAL DASHBOARD")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_path}")
    print(f"\nSteps:")
    print(f"  1. Probe preprocessing: {'SKIP' if args.skip_probes else 'RUN'}")
    print(f"  2. Logit lens:          {'SKIP' if args.skip_logit else 'RUN'}")
    print(f"  3. Axis lens:           {'SKIP' if args.skip_axis else 'RUN'}")

    # Validate input
    if not args.skip_probes and not input_path.exists():
        print(f"\nERROR: Input file not found: {input_path}")
        return 1

    if args.skip_probes and not output_path.exists():
        print(f"\nERROR: Cannot skip probes - output file doesn't exist: {output_path}")
        return 1

    # Execute pipeline
    success = True

    # Step 1: Probe preprocessing
    if not args.skip_probes:
        if not run_probe_preprocessing(input_path, output_path, args.probes):
            success = False
            print("\nERROR: Probe preprocessing failed, aborting pipeline")
            return 1
    else:
        print_header("SKIPPING PROBE PREPROCESSING", "1/3")
        print(f"Using existing pickle: {output_path}\n")

    # Step 2: Logit lens
    if not args.skip_logit:
        if not run_logit_lens_preprocessing(output_path):
            success = False
            print("\nWARNING: Logit lens failed, but continuing...")
    else:
        print_header("SKIPPING LOGIT LENS PREPROCESSING", "2/3")

    # Step 3: Axis lens
    if not args.skip_axis:
        if not run_axis_lens_preprocessing(output_path):
            success = False
            print("\nWARNING: Axis lens failed, but continuing...")
    else:
        print_header("SKIPPING AXIS LENS PREPROCESSING", "3/3")

    # Final summary
    print("\n" + "="*80)
    if success:
        print("PREPROCESSING COMPLETE!")
    else:
        print("PREPROCESSING COMPLETED WITH WARNINGS")
    print("="*80)

    print(f"\nOutput file: {output_path}")
    if output_path.exists():
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"File size: {size_mb:.1f} MB")

    print("\nProbe types included:")
    if not args.skip_probes:
        print("  Standard probes:")
        for probe in args.probes:
            print(f"    - {probe}")

    if not args.skip_logit:
        print("  Logit lens:")
        print("    - logit_lens_mean (L40-50)")
        print("    - logit_lens_mean_l30_40 (L30-40)")
        print("    - logit_lens_mean_l20_30 (L20-30)")

    if not args.skip_axis:
        print("  Axis lens:")
        print("    - axis_lens_mean")

    print("\nFeatures:")
    if not args.skip_logit:
        print("  ✓ Baseline correction applied to logit lens")
        print("  ✓ Per-layer scores stored for layerwise plotting")
    if not args.skip_axis:
        print("  ✓ Baseline correction applied to axis lens")
    print("  ✓ All probes computed in optimal order")

    print("\n" + "="*80)

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
