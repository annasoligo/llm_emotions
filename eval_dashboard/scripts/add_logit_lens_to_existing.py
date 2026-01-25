#!/usr/bin/env python3
"""
Add logit lens results to existing preprocessed dashboard data.
Runs preprocessing for ONLY logit lens probes, then merges into existing file.
"""

import sys
sys.path.insert(0, '/workspace-vast/annas/git/research-tools')

import pickle
from pathlib import Path
import subprocess

# Paths
EXISTING_FILE = Path('/workspace-vast/annas/git/research-tools/eval_dashboard/data/high_emotion_6plus.pkl')
TEMP_FILE = Path('/workspace-vast/annas/git/research-tools/eval_dashboard/data/logit_lens_temp.pkl')
INPUT_DATA = Path('/workspace-vast/annas/git/research-tools/elicitation/outputs/dashboard_subsets/high_emotion_6plus.jsonl')

print("="*80)
print("ADDING LOGIT LENS TO EXISTING DASHBOARD DATA")
print("="*80)
print()

# Step 1: Run preprocessing with ONLY logit lens probes
print("[1/3] Computing logit lens scores...")
cmd = [
    'python', 'eval_dashboard/data_preprocessing.py',
    '--input', str(INPUT_DATA),
    '--output', str(TEMP_FILE),
    '--probes', 'logit_lens_mean', 'logit_lens_max'
]
result = subprocess.run(cmd, capture_output=False)
if result.returncode != 0:
    print(f"Error running preprocessing: {result.returncode}")
    sys.exit(1)

print()
print("[2/3] Merging with existing data...")

# Load existing data
with open(EXISTING_FILE, 'rb') as f:
    existing_data = pickle.load(f)

# Load new logit lens data
with open(TEMP_FILE, 'rb') as f:
    new_data = pickle.load(f)

# Merge probe_configs
print(f"  Merging probe configs...")
existing_data['probe_configs'].update(new_data['probe_configs'])
print(f"  ✓ Added {len(new_data['probe_configs'])} new probe configs")

# Merge probe_baselines
print(f"  Merging probe baselines...")
existing_data['probe_baselines'].update(new_data['probe_baselines'])
print(f"  ✓ Added {len(new_data['probe_baselines'])} new probe baselines")

# Merge probe_scores
print(f"  Existing conversations: {len(existing_data['conversations'])}")
print(f"  Logit lens conversations: {len(new_data['conversations'])}")

merged_count = 0
for existing_conv, new_conv in zip(existing_data['conversations'], new_data['conversations']):
    # Verify they're the same conversation
    if existing_conv['sample_id'] != new_conv['sample_id']:
        print(f"  WARNING: Mismatched sample IDs: {existing_conv['sample_id']} vs {new_conv['sample_id']}")
        continue

    # Merge probe_scores
    existing_conv['probe_scores'].update(new_conv['probe_scores'])
    merged_count += 1

print(f"  ✓ Merged {merged_count} conversations")

# Step 3: Save updated data
print()
print("[3/3] Saving updated data...")
with open(EXISTING_FILE, 'wb') as f:
    pickle.dump(existing_data, f)

# Clean up temp file
TEMP_FILE.unlink()

print(f"  ✓ Saved to {EXISTING_FILE}")
print()
print("="*80)
print("SUCCESS! Logit lens added to dashboard data")
print("="*80)
print()

# Show final probe list
if existing_data['conversations']:
    probes = list(existing_data['conversations'][0]['probe_scores'].keys())
    print("Probes now available:")
    for p in probes:
        print(f"  - {p}")
