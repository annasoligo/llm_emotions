#!/usr/bin/env python3
"""Wrapper to generate batch 2 samples that append to existing files."""

import subprocess
import sys
import tempfile
import shutil
from pathlib import Path

def main():
    if len(sys.argv) < 4:
        print("Usage: generate_batch2.py <condition> <scenario> <entity> [--base-model]")
        sys.exit(1)

    condition = sys.argv[1]
    scenario = sys.argv[2]
    entity = sys.argv[3]
    is_base = "--base-model" in sys.argv

    # Determine script and output file
    if is_base:
        script = "generate_entity_trials_basemodel.py"
        prefix = "basemodel_"
    else:
        script = "generate_entity_trials.py"
        prefix = ""

    output_file = Path("outputs") / f"{prefix}{scenario}_{entity}_{condition}.jsonl"

    print(f"Generating batch 2 for: {condition} / {scenario} / {entity}")
    print(f"Output file: {output_file}")

    # Check if original file exists
    if not output_file.exists():
        print(f"ERROR: Original file not found: {output_file}")
        print("Cannot append to non-existent file!")
        sys.exit(1)

    # Count existing samples
    with open(output_file) as f:
        existing_count = sum(1 for _ in f)
    print(f"Existing samples: {existing_count}")

    # Generate to temporary file
    temp_file = output_file.parent / f"_temp_{output_file.name}"

    # Temporarily rename output file so generation script creates new one
    backup_file = output_file.parent / f"_backup_{output_file.name}"
    shutil.copy(output_file, backup_file)

    try:
        # Run generation (will create fresh file)
        cmd = ["python", script, condition, scenario, entity, "--num-samples", "30"]
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True)

        # Now append new file to backup
        with open(output_file, 'r') as new_f:
            with open(backup_file, 'a') as orig_f:
                for line in new_f:
                    orig_f.write(line)

        # Replace original with merged version
        shutil.move(backup_file, output_file)

        # Count final samples
        with open(output_file) as f:
            final_count = sum(1 for _ in f)
        print(f"Final samples: {final_count} (added {final_count - existing_count})")

    except Exception as e:
        print(f"ERROR: {e}")
        # Restore from backup
        if backup_file.exists():
            shutil.move(backup_file, output_file)
        raise
    finally:
        # Clean up
        if backup_file.exists():
            backup_file.unlink()

if __name__ == "__main__":
    main()
