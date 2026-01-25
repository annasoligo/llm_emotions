"""
Copy orthogonal_regularized scores from preprocessed_conversations.pkl to subset files.
Much faster than recomputing!
"""
import pickle
from pathlib import Path

print("="*80)
print("COPYING ORTHOGONAL REGULARIZED SCORES TO SUBSET FILES")
print("="*80)

data_dir = Path("/workspace-vast/annas/git/research-tools/eval_dashboard/data")
probe_key = 'orthogonal_regularized_lambda100'

# Load source data with orthogonal_regularized scores
print("\nLoading source data (preprocessed_conversations.pkl)...")
with open(data_dir / 'preprocessed_conversations.pkl', 'rb') as f:
    source_data = pickle.load(f)

# Build lookup: sample_id -> orthogonal_regularized scores
print("Building score lookup by sample_id...")
score_lookup = {}
for conv in source_data['conversations']:
    sample_id = conv['sample_id']
    if probe_key in conv['probe_scores']:
        score_lookup[sample_id] = conv['probe_scores'][probe_key]

print(f"✓ Found {len(score_lookup)} conversations with {probe_key} scores")

# Get probe config and baseline
probe_config = source_data['probe_configs'][probe_key]
probe_baseline = source_data['probe_baselines'][probe_key]

# Process each subset file
subset_files = [
    'high_emotion_6plus.pkl',
    'mid_emotion_3to5.pkl',
    'low_emotion_0to2.pkl',
    'low_emotion_with_shutdown.pkl',
    'low_emotion_no_shutdown.pkl',
    'baseline_v12_solvable.pkl'
]

for subset_file in subset_files:
    file_path = data_dir / subset_file

    if not file_path.exists():
        print(f"\n⚠ Skipping {subset_file} - file not found")
        continue

    print(f"\n{'='*80}")
    print(f"Processing: {subset_file}")
    print(f"{'='*80}")

    # Load subset data
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    conversations = data['conversations']
    print(f"  Loaded {len(conversations)} conversations")

    # Copy scores by matching sample_id
    matched = 0
    skipped = 0
    for conv in conversations:
        sample_id = conv['sample_id']

        if sample_id in score_lookup:
            # Copy the scores
            conv['probe_scores'][probe_key] = score_lookup[sample_id]
            matched += 1
        else:
            print(f"  ⚠ Warning: No scores found for sample_id {sample_id}")
            skipped += 1

    print(f"  ✓ Copied scores for {matched} conversations")
    if skipped > 0:
        print(f"  ⚠ Skipped {skipped} conversations (no scores found)")

    # Update metadata
    if 'probe_configs' not in data:
        data['probe_configs'] = {}
    data['probe_configs'][probe_key] = probe_config

    if 'probe_baselines' not in data:
        data['probe_baselines'] = {}
    data['probe_baselines'][probe_key] = probe_baseline

    # Save updated data
    print(f"  Saving {subset_file}...")
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

    print(f"  ✓ {subset_file} updated successfully")

print("\n" + "="*80)
print("DONE!")
print("="*80)
print(f"\nOrthogonal regularized scores copied to all subset files.")
print(f"Reload the dashboard (click 'Reload All Data' button) to see 'Ortho Reg λ=100'.")
