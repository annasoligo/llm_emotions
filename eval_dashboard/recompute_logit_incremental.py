"""
Recompute logit_lens data with incremental checkpoints for verification.
Saves intermediate results every 10 conversations.
"""
import sys
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))

from eval_dashboard.data_preprocessing import preprocess_all_conversations
from eval_dashboard.probe_configs import PROBE_CONFIGS

EMOTIONS = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']

def compute_correlation_stats(conversations):
    """Compute correlation statistics for verification."""
    all_scores = []

    for conv in conversations:
        if 'probe_scores' not in conv or 'logit_lens_mean' not in conv['probe_scores']:
            continue

        for sent_id, scores in conv['probe_scores']['logit_lens_mean'].items():
            if isinstance(scores, np.ndarray) and len(scores) == 6:
                all_scores.append(scores)

    if not all_scores:
        return None

    scores_matrix = np.array(all_scores)
    corr_matrix = np.corrcoef(scores_matrix.T)

    # Compute average absolute correlation
    off_diag_corrs = []
    for i in range(6):
        for j in range(i+1, 6):
            off_diag_corrs.append(abs(corr_matrix[i, j]))

    return {
        'n_scores': len(all_scores),
        'mean': scores_matrix.mean(),
        'std': scores_matrix.std(),
        'min': scores_matrix.min(),
        'max': scores_matrix.max(),
        'avg_correlation': np.mean(off_diag_corrs),
        'corr_matrix': corr_matrix
    }

def main():
    print("="*80)
    print("INCREMENTAL LOGIT LENS RECOMPUTATION")
    print("="*80)

    # Paths
    input_path = Path('/workspace-vast/annas/git/research-tools/eval_dashboard/data/high_emotion_6plus.pkl')
    output_path = Path('/workspace-vast/annas/git/research-tools/eval_dashboard/data/high_emotion_6plus_logit_fixed.pkl')
    checkpoint_dir = Path('/workspace-vast/annas/git/research-tools/eval_dashboard/checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)

    print(f"\nInput:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Checkpoints: {checkpoint_dir}")

    # Load existing data
    print("\n[1/3] Loading existing dashboard data...")
    with open(input_path, 'rb') as f:
        data = pickle.load(f)

    print(f"  Total conversations: {len(data['conversations'])}")
    print(f"  Existing probes: {list(data['probe_configs'].keys())}")

    # Process in batches of 10
    BATCH_SIZE = 10
    total_convs = len(data['conversations'])
    num_batches = (total_convs + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"\n[2/3] Processing {total_convs} conversations in {num_batches} batches of {BATCH_SIZE}...")

    # Load original jsonl for reprocessing
    jsonl_path = Path('/workspace-vast/annas/git/research-tools/eval_dashboard/data/high_emotion_6plus_with_turns.jsonl')

    all_processed = []

    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, total_convs)

        print(f"\n{'='*80}")
        print(f"BATCH {batch_idx+1}/{num_batches}: Conversations {start_idx}-{end_idx-1}")
        print(f"{'='*80}")

        # Get conversation IDs for this batch
        batch_conv_ids = [data['conversations'][i]['id'] for i in range(start_idx, end_idx)]

        # Process this batch
        batch_data = preprocess_all_conversations(
            jsonl_path=jsonl_path,
            probe_keys=['logit_lens_mean', 'logit_lens_max'],
            model_name='google/gemma-3-27b-it',
            device='cuda:0',
            conversation_ids=batch_conv_ids,
            skip_baseline_computation=True  # Skip WildChat baseline
        )

        all_processed.extend(batch_data['conversations'])

        # Compute statistics for verification
        print(f"\n{'='*80}")
        print(f"VERIFICATION - Batch {batch_idx+1}")
        print(f"{'='*80}")

        stats = compute_correlation_stats(batch_data['conversations'])

        if stats:
            print(f"\nProcessed {stats['n_scores']} sentence scores")
            print(f"Score statistics:")
            print(f"  Mean: {stats['mean']:.3f}")
            print(f"  Std:  {stats['std']:.3f}")
            print(f"  Min:  {stats['min']:.3f}")
            print(f"  Max:  {stats['max']:.3f}")
            print(f"\nAverage correlation: {stats['avg_correlation']:.3f}")
            print(f"Expected: ~0.5-0.6 (moderate correlation like emo_lens)")

            if stats['avg_correlation'] > 0.8:
                print("⚠ WARNING: Very high correlation! Check if baseline is loading correctly")
            elif stats['avg_correlation'] < 0.3:
                print("⚠ WARNING: Very low correlation! This is unexpected")
            else:
                print("✓ Correlation looks reasonable")

            if abs(stats['mean']) > 10:
                print(f"⚠ WARNING: Mean is {stats['mean']:.1f}, should be ~0! Baseline may not be applied")
            else:
                print(f"✓ Mean is {stats['mean']:.3f}, close to 0 (good z-score normalization)")

        # Save checkpoint
        checkpoint_path = checkpoint_dir / f"checkpoint_batch_{batch_idx+1:03d}.pkl"

        # Update conversation data with new logit_lens scores
        updated_conversations = data['conversations'][:start_idx] + all_processed

        checkpoint_data = {
            'conversations': updated_conversations,
            'probe_configs': {
                **data['probe_configs'],
                'logit_lens_mean': PROBE_CONFIGS['logit_lens_mean'],
                'logit_lens_max': PROBE_CONFIGS['logit_lens_max']
            },
            'emotions': EMOTIONS,
            'batch_info': {
                'completed_batches': batch_idx + 1,
                'total_batches': num_batches,
                'conversations_processed': len(all_processed)
            }
        }

        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)

        print(f"\n✓ Checkpoint saved: {checkpoint_path}")
        print(f"  Progress: {len(all_processed)}/{total_convs} conversations ({100*len(all_processed)/total_convs:.1f}%)")

    # Save final output
    print(f"\n{'='*80}")
    print("[3/3] Saving final output...")
    print(f"{'='*80}")

    # Combine old conversations with new logit_lens scores
    final_conversations = []
    for old_conv, new_conv in zip(data['conversations'], all_processed):
        # Merge probe scores
        updated_conv = old_conv.copy()
        if 'probe_scores' not in updated_conv:
            updated_conv['probe_scores'] = {}

        # Add new logit_lens scores
        for probe_key in ['logit_lens_mean', 'logit_lens_max']:
            if probe_key in new_conv.get('probe_scores', {}):
                updated_conv['probe_scores'][probe_key] = new_conv['probe_scores'][probe_key]

        final_conversations.append(updated_conv)

    final_data = {
        'conversations': final_conversations,
        'probe_configs': {
            **data['probe_configs'],
            'logit_lens_mean': PROBE_CONFIGS['logit_lens_mean'],
            'logit_lens_max': PROBE_CONFIGS['logit_lens_max']
        },
        'emotions': EMOTIONS
    }

    with open(output_path, 'wb') as f:
        pickle.dump(final_data, f)

    print(f"✓ Saved final output: {output_path}")
    print(f"  Size: {output_path.stat().st_size / 1e6:.1f} MB")

    # Final statistics
    print(f"\n{'='*80}")
    print("FINAL VERIFICATION")
    print(f"{'='*80}")

    final_stats = compute_correlation_stats(final_conversations)

    if final_stats:
        print(f"\nTotal: {final_stats['n_scores']} sentence scores")
        print(f"\nScore statistics:")
        print(f"  Mean: {final_stats['mean']:.3f}")
        print(f"  Std:  {final_stats['std']:.3f}")
        print(f"  Min:  {final_stats['min']:.3f}")
        print(f"  Max:  {final_stats['max']:.3f}")
        print(f"\nCorrelation matrix:")
        print("           ", " ".join(f"{e[:6]:>7s}" for e in EMOTIONS))
        for i, emotion in enumerate(EMOTIONS):
            row = ' '.join(f'{final_stats["corr_matrix"][i, j]:7.3f}' for j in range(6))
            print(f"  {emotion:10s} {row}")
        print(f"\nAverage absolute correlation: {final_stats['avg_correlation']:.3f}")

        if final_stats['avg_correlation'] < 0.7 and abs(final_stats['mean']) < 2:
            print("\n✓✓✓ SUCCESS! Scores look correct!")
        else:
            print("\n⚠⚠⚠ WARNING: Check the results carefully")

    print(f"\n{'='*80}")
    print("DONE!")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()
