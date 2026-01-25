"""
Recompute logit_lens data with incremental checkpoints - simplified version.
Processes first 20 conversations as a test, saves intermediate results.
"""
import sys
import json
import pickle
import numpy as np
from pathlib import Path

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
    print("INCREMENTAL LOGIT LENS RECOMPUTATION - TEST RUN")
    print("="*80)

    # Paths
    input_jsonl = Path('/workspace-vast/annas/git/research-tools/eval_dashboard/data/high_emotion_6plus_with_turns.jsonl')
    test_jsonl = Path('/tmp/test_first_20.jsonl')
    test_output = Path('/workspace-vast/annas/git/research-tools/eval_dashboard/data/test_first_20_logit.pkl')

    print(f"\nProcessing first 20 conversations as TEST")
    print(f"Input:  {input_jsonl}")
    print(f"Output: {test_output}")

    # Create test JSONL with first 20 conversations
    print("\n[1/3] Creating test dataset...")
    with open(input_jsonl) as f_in, open(test_jsonl, 'w') as f_out:
        for i, line in enumerate(f_in):
            if i >= 20:
                break
            f_out.write(line)

    print(f"✓ Created test dataset: {test_jsonl}")

    # Process with logit_lens
    print("\n[2/3] Processing with logit_lens...")
    print("This will take ~10-15 minutes for 20 conversations")
    print("Watch for:")
    print("  - 'Computing layer-averaged baseline stats' (shows baseline loading)")
    print("  - '✓ Averaged stats for 1231 tokens' (correct baseline)")
    print("  - Score statistics after processing")

    preprocess_all_conversations(
        data_path=str(test_jsonl),
        probe_keys=['logit_lens_mean'],
        output_path=str(test_output),
        use_simple_splitter=True
    )

    # Verify results
    print(f"\n{'='*80}")
    print("[3/3] VERIFICATION")
    print(f"{'='*80}")

    with open(test_output, 'rb') as f:
        data = pickle.load(f)

    print(f"\nLoaded {len(data['conversations'])} conversations")

    stats = compute_correlation_stats(data['conversations'])

    if stats:
        print(f"\nProcessed {stats['n_scores']} sentence scores total")
        print(f"\nScore statistics:")
        print(f"  Mean: {stats['mean']:.3f}")
        print(f"  Std:  {stats['std']:.3f}")
        print(f"  Min:  {stats['min']:.3f}")
        print(f"  Max:  {stats['max']:.3f}")

        print(f"\nCorrelation matrix:")
        print("           ", " ".join(f"{e[:6]:>7s}" for e in EMOTIONS))
        for i, emotion in enumerate(EMOTIONS):
            row = ' '.join(f'{stats["corr_matrix"][i, j]:7.3f}' for j in range(6))
            print(f"  {emotion:10s} {row}")

        print(f"\nAverage absolute correlation: {stats['avg_correlation']:.3f}")
        print(f"Expected: ~0.5-0.6 (moderate, like emo_lens)")

        # Validation
        print(f"\n{'='*80}")
        print("VALIDATION")
        print(f"{'='*80}")

        passed = True

        if abs(stats['mean']) > 5:
            print(f"✗ FAIL: Mean is {stats['mean']:.1f}, should be ~0")
            print("  → Baseline normalization not being applied!")
            passed = False
        else:
            print(f"✓ PASS: Mean is {stats['mean']:.3f} (z-score normalized)")

        if stats['avg_correlation'] > 0.8:
            print(f"⚠ WARNING: Correlation is {stats['avg_correlation']:.3f}, higher than expected")
            print("  → May indicate baseline issues, but emo_lens also shows moderate correlation")
            # Don't fail on this since emo_lens itself shows 0.56
        elif stats['avg_correlation'] < 0.3:
            print(f"⚠ WARNING: Correlation is {stats['avg_correlation']:.3f}, lower than emo_lens")
            print("  → Unexpected, emo_lens shows ~0.56")
        else:
            print(f"✓ PASS: Correlation is {stats['avg_correlation']:.3f} (moderate, similar to emo_lens)")

        if passed:
            print(f"\n{'='*80}")
            print("✓✓✓ TEST PASSED! ✓✓✓")
            print(f"{'='*80}")
            print("\nReady to run full recomputation on all conversations")
        else:
            print(f"\n{'='*80}")
            print("✗✗✗ TEST FAILED ✗✗✗")
            print(f"{'='*80}")
            print("\nDO NOT run full recomputation until this is fixed")

    print(f"\n{'='*80}")
    print("DONE!")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()
