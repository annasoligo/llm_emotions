"""
Remove orthogonal_regularized scores from all subset files.
"""
import pickle
from pathlib import Path

data_dir = Path("/workspace-vast/annas/git/research-tools/eval_dashboard/data")
probe_key = 'orthogonal_regularized_lambda100'

subset_files = [
    'high_emotion_6plus.pkl',
    'mid_emotion_3to5.pkl',
    'low_emotion_0to2.pkl',
    'low_emotion_with_shutdown.pkl',
    'low_emotion_no_shutdown.pkl',
    'baseline_v12_solvable.pkl'
]

print("Removing orthogonal_regularized scores from subset files...")

for subset_file in subset_files:
    file_path = data_dir / subset_file

    if not file_path.exists():
        continue

    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    # Remove scores from conversations
    removed = 0
    for conv in data['conversations']:
        if probe_key in conv['probe_scores']:
            del conv['probe_scores'][probe_key]
            removed += 1

    # Remove from metadata
    if 'probe_configs' in data and probe_key in data['probe_configs']:
        del data['probe_configs'][probe_key]
    if 'probe_baselines' in data and probe_key in data['probe_baselines']:
        del data['probe_baselines'][probe_key]

    # Save
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

    print(f"  ✓ {subset_file}: Removed from {removed} conversations")

print("\n✓ All orthogonal_regularized scores removed")
