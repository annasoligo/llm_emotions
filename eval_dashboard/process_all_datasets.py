"""
Process all eval_dashboard datasets to add axis_lens scores.
Runs the preprocessing on multiple pickle files in sequence.
"""
import sys
from pathlib import Path
import subprocess

# List of datasets to process
DATASETS = [
    'high_emotion_6plus.pkl',
    'mid_emotion_3to5.pkl',
    'low_emotion_0to2.pkl',
    'low_emotion_no_shutdown.pkl',
    'low_emotion_with_shutdown.pkl',
    'baseline_v12_solvable.pkl'
]

DATA_DIR = Path('/workspace-vast/annas/git/research-tools/eval_dashboard/data')

def main():
    print("="*80)
    print("Processing All Datasets with Axis Scores")
    print("="*80)

    for i, dataset in enumerate(DATASETS, 1):
        input_path = DATA_DIR / dataset

        # Skip if doesn't exist
        if not input_path.exists():
            print(f"\n[{i}/{len(DATASETS)}] SKIP: {dataset} (not found)")
            continue

        # Generate output path
        output_name = dataset.replace('.pkl', '_with_axes.pkl')
        output_path = DATA_DIR / output_name

        # Skip if already processed
        if output_path.exists():
            print(f"\n[{i}/{len(DATASETS)}] SKIP: {dataset} (already has axes)")
            continue

        print(f"\n[{i}/{len(DATASETS)}] Processing: {dataset}")
        print(f"  Input:  {input_path}")
        print(f"  Output: {output_path}")

        # Create a temporary script for this dataset
        script_content = f"""
import sys
from pathlib import Path
sys.path.insert(0, "/workspace-vast/annas/git/believe-it-or-not")
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import preprocessing function
exec(open('/workspace-vast/annas/git/research-tools/eval_dashboard/add_axis_lens_to_existing.py').read())

# Override paths
input_path = Path('{input_path}')
output_path = Path('{output_path}')

# Run with modified paths
import pickle
import numpy as np
from emotion_evals.emo_lens.token_trajectories import (
    extract_token_level_activations,
    compute_layer_averaged_axis_baseline_stats,
    compute_token_axis_scores
)
from emotion_evals.emo_lens.model_utils import load_base_model
from emotion_evals.emo_lens.logit_lens_emotion_direct import (
    load_reference_stats_for_strategy,
    load_axis_token_groups
)

AXES = ['valence', 'arousal', 'dominance', 'approach_avoidance']

print("="*80)
print(f"Processing: {input_path.name}")
print("="*80)
print(f"Input:  {{input_path}}")
print(f"Output: {{output_path}}")

# Load data
print("\\n[1/5] Loading existing data...")
with open(input_path, 'rb') as f:
    data = pickle.load(f)
total_convs = len(data['conversations'])
print(f"  Total conversations: {{total_convs}}")

# Load model
print("\\n[2/5] Loading model...")
BASE_MODEL_NAME = "unsloth/gemma-3-27b-it"
model, tokenizer = load_base_model(BASE_MODEL_NAME)
print("  ✓ Model loaded")

# Load axis token groups
print("\\n[3/5] Loading axis token groups...")
axis_token_groups = load_axis_token_groups(BASE_MODEL_NAME)
print(f"  ✓ Loaded {{len(axis_token_groups)}} axes")

# Load baseline stats
print("\\n[4/5] Loading baseline statistics...")
ACTIVATION_STRATEGY = "generated_tokens_avg"
SCRIPT_DIR = Path("/workspace-vast/annas/git/believe-it-or-not/emotion_evals/emo_lens")
ref_stats = load_reference_stats_for_strategy(
    model_name=BASE_MODEL_NAME,
    activation_strategy=ACTIVATION_STRATEGY,
    script_dir=SCRIPT_DIR
)
print("  ✓ Loaded baseline stats")

LAYER_RANGE = list(range(40, 51))
averaged_axis_baseline_stats = compute_layer_averaged_axis_baseline_stats(
    ref_stats=ref_stats,
    layers=LAYER_RANGE,
    axis_token_groups=axis_token_groups
)
print("  ✓ Computed averaged axis baseline statistics")

# Process conversations
print(f"\\n[5/5] Processing {{total_convs}} conversations...")
for i in range(total_convs):
    conv = data['conversations'][i]
    print(f"  [{{i+1}}/{{total_convs}}] Sample ID: {{conv['sample_id']}}")

    # Build conversation text
    conversation_text = ""
    for turn in conv['conversation']:
        role = turn['role']
        content = turn['content']
        if role == 'user':
            conversation_text += f"User: {{content}}\\n"
        else:
            conversation_text += f"Assistant: {{content}}\\n"

    # Extract activations
    activations_by_token, token_ids = extract_token_level_activations(
        model=model,
        tokenizer=tokenizer,
        prompt=conversation_text,
        layers=LAYER_RANGE,
        start_token_idx=0,
        system_prompt=None,
        num_generated_tokens=0
    )

    # Compute axis scores
    token_axis_scores = compute_token_axis_scores(
        model=model,
        activations_by_token=activations_by_token,
        layers=LAYER_RANGE,
        axis_token_groups=axis_token_groups,
        averaged_axis_baseline_stats=averaged_axis_baseline_stats,
        aggregation="mean",
        subtract_mean=True
    )

    # Aggregate to sentence level
    sentence_scores = {{}}
    for sent in conv['sentences']:
        sent_id = sent['sentence_id']
        start_tok = sent['start_token']
        end_tok = sent['end_token']

        sent_score_list = []
        for t in range(start_tok, end_tok):
            if t in token_axis_scores:
                scores_array = np.array([token_axis_scores[t][a] for a in AXES])
                sent_score_list.append(scores_array)

        if sent_score_list:
            sentence_scores[sent_id] = np.mean(sent_score_list, axis=0)
        else:
            sentence_scores[sent_id] = np.zeros(4)

    conv['probe_scores']['axis_lens_mean'] = sentence_scores
    print(f"    ✓ Computed scores for {{len(sentence_scores)}} sentences")

# Update probe configs
from eval_dashboard.probe_configs import PROBE_CONFIGS
data['probe_configs']['axis_lens_mean'] = PROBE_CONFIGS['axis_lens_mean']

# Save
print(f"\\nSaving to {{output_path}}...")
with open(output_path, 'wb') as f:
    pickle.dump(data, f)
print(f"✓ Saved: {{output_path}}")
print(f"  Size: {{output_path.stat().st_size / 1e6:.1f}} MB")
"""

        # Write temporary script
        temp_script = DATA_DIR / f'temp_process_{dataset}.py'
        with open(temp_script, 'w') as f:
            f.write(script_content)

        print(f"  Submitting job...")
        # Note: Would need to submit as SLURM job, but for now just flag it
        print(f"  ⚠ Run this manually or via SLURM")

    print(f"\n{'='*80}")
    print("Summary:")
    print(f"{'='*80}")
    print(f"Total datasets: {len(DATASETS)}")
    print("\nTo process these datasets, create SLURM jobs for each.")

if __name__ == '__main__':
    main()
