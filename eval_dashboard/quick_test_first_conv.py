"""Quick test to validate the fix on just the first conversation."""
import pickle
import numpy as np
import sys
from pathlib import Path

# Add paths
sys.path.insert(0, "/workspace-vast/annas/git/believe-it-or-not")
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import main preprocessing function
from add_logit_lens_to_existing_BATCHED import (
    EMOTIONS, LAYER_RANGE_CONFIGS
)

# Load backup data
backup_path = Path("/workspace-vast/annas/git/research-tools/eval_dashboard/data/high_emotion_6plus_with_axes.pkl.backup_before_layerwise")
print("Loading backup data...")
with open(backup_path, 'rb') as f:
    backup_data = pickle.load(f)

# Load current full run (if exists)
current_path = Path("/workspace-vast/annas/git/research-tools/eval_dashboard/data/high_emotion_6plus_with_layerwise.pkl")
if current_path.exists():
    print("Loading current data...")
    with open(current_path, 'rb') as f:
        current_data = pickle.load(f)

    # Compare first few sentences from first conversation
    conv_0_backup = backup_data['conversations'][0]
    conv_0_current = current_data['conversations'][0]

    probe_key = 'logit_lens_mean'

    print(f"\nComparing '{probe_key}' scores:")
    print("="*80)

    # Get first 5 sentences that have this probe
    count = 0
    max_diff_list = []

    for sent_id, sent_data_backup in conv_0_backup['sentences'].items():
        if probe_key not in sent_data_backup:
            continue

        sent_data_current = conv_0_current['sentences'][sent_id]

        backup_scores = np.array(sent_data_backup[probe_key])
        current_scores = np.array(sent_data_current[probe_key])

        diff = np.abs(backup_scores - current_scores)
        max_diff = np.max(diff)
        max_diff_list.append(max_diff)

        if count < 5:
            print(f"\nSentence {sent_id}:")
            print(f"  BACKUP:  {backup_scores}")
            print(f"  CURRENT: {current_scores}")
            print(f"  DIFF:    {diff}")
            print(f"  MAX DIFF: {max_diff:.6f}")

        count += 1
        if count >= 10:
            break

    print(f"\n" + "="*80)
    print(f"Statistics across {len(max_diff_list)} sentences:")
    print(f"  Mean max diff: {np.mean(max_diff_list):.6f}")
    print(f"  Median max diff: {np.median(max_diff_list):.6f}")
    print(f"  Max diff overall: {np.max(max_diff_list):.6f}")
    print(f"  Min diff overall: {np.min(max_diff_list):.6f}")

    if np.mean(max_diff_list) < 0.001:
        print("\n✓ FIX SUCCESSFUL! Scores match backup within tolerance.")
    elif np.mean(max_diff_list) < 0.01:
        print("\n⚠ Scores are close but have small differences (< 0.01)")
    else:
        print("\n✗ FIX INCOMPLETE! Scores still differ significantly.")

else:
    print("Current data file doesn't exist yet - job still running.")
    print("Will need to wait for job to complete before comparing.")
