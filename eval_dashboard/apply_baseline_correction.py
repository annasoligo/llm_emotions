"""
Apply baseline correction to logit_lens emotion scores AND axis scores.

Removes the component of scores that correlates with overall logit magnitude,
leaving only emotion/axis-specific variation independent of general confidence.
"""
import pickle
import numpy as np
from pathlib import Path

def apply_baseline_correction(input_path, output_path=None):
    """
    Apply baseline correction to emotion scores AND axis scores.

    Computes: corrected_score = score - alpha * normalized_baseline
    Where alpha is chosen to make corrected_score independent of baseline.

    Applies to both:
    - logit_lens_mean (6 emotions)
    - axis_lens_mean (4 axes)
    """
    if output_path is None:
        output_path = input_path

    print("="*80)
    print("BASELINE CORRECTION FOR EMOTION SCORES")
    print("="*80)
    print(f"\nInput:  {input_path}")
    print(f"Output: {output_path}")

    # Load data
    print("\n[1/4] Loading data...")
    with open(input_path, 'rb') as f:
        data = pickle.load(f)

    print(f"  Conversations: {len(data['conversations'])}")

    # Collect all emotion scores and raw mean logits
    print("\n[2/4] Collecting scores...")
    all_emotion_scores = []
    all_raw_logits = []
    all_sentence_ids = []  # Track (conv_idx, sent_id) for updating later

    for conv_idx, conv in enumerate(data['conversations']):
        sent_logits = conv['sentence_mean_logits'].get('axis_lens_mean', {})
        emotion_scores_dict = conv['probe_scores'].get('logit_lens_mean', {})

        for sent_id in sent_logits.keys():
            if sent_id in emotion_scores_dict:
                raw_logit = sent_logits[sent_id]
                emotion_score = np.array(emotion_scores_dict[sent_id])

                all_raw_logits.append(raw_logit)
                all_emotion_scores.append(emotion_score)
                all_sentence_ids.append((conv_idx, sent_id))

    all_raw_logits = np.array(all_raw_logits)
    all_emotion_scores = np.array(all_emotion_scores)  # (n_sentences, 6)

    print(f"  Collected {len(all_raw_logits)} sentence scores")

    # Compute correction for each emotion
    print("\n[3/4] Computing corrections...")

    # Normalize raw logits
    raw_normalized = (all_raw_logits - all_raw_logits.mean()) / all_raw_logits.std()

    # Compute correction for each emotion dimension separately
    corrections = {}
    for emotion_idx, emotion in enumerate(['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']):
        emotion_scores = all_emotion_scores[:, emotion_idx]

        # Compute correlation
        corr = np.corrcoef(emotion_scores, all_raw_logits)[0, 1]

        # Compute optimal alpha to remove correlation
        alpha = corr * emotion_scores.std() / raw_normalized.std()

        corrections[emotion] = {
            'alpha': alpha,
            'correlation_before': corr
        }

        print(f"  {emotion:10s}: corr={corr:6.3f}, alpha={alpha:6.3f}")

    # Also compute correction for mean emotion score
    mean_emotion_scores = all_emotion_scores.mean(axis=1)
    corr_mean = np.corrcoef(mean_emotion_scores, all_raw_logits)[0, 1]
    alpha_mean = corr_mean * mean_emotion_scores.std() / raw_normalized.std()

    print(f"\n  {'mean':10s}: corr={corr_mean:6.3f}, alpha={alpha_mean:6.3f}")

    # Apply corrections
    print("\n[4/4] Applying corrections...")

    # Use the mean alpha for all emotions (simpler and more stable)
    correction_vector = alpha_mean * raw_normalized

    n_corrected = 0
    for idx, (conv_idx, sent_id) in enumerate(all_sentence_ids):
        conv = data['conversations'][conv_idx]

        # Get original scores
        original_scores = np.array(conv['probe_scores']['logit_lens_mean'][sent_id])

        # Apply correction (same correction to all emotions)
        corrected_scores = original_scores - correction_vector[idx]

        # Update in place
        conv['probe_scores']['logit_lens_mean'][sent_id] = corrected_scores.tolist()
        n_corrected += 1

    print(f"  ✓ Corrected {n_corrected} emotion sentences")

    # =========================================================================
    # AXIS SCORES CORRECTION
    # =========================================================================
    print("\n" + "="*80)
    print("CORRECTING AXIS SCORES")
    print("="*80)

    # Collect all axis scores and raw mean logits
    print("\n[1/3] Collecting axis scores...")
    all_axis_scores = []
    all_axis_raw_logits = []
    all_axis_sentence_ids = []  # Track (conv_idx, sent_id) for updating later

    for conv_idx, conv in enumerate(data['conversations']):
        sent_logits = conv['sentence_mean_logits'].get('axis_lens_mean', {})
        axis_scores_dict = conv['probe_scores'].get('axis_lens_mean', {})

        for sent_id in sent_logits.keys():
            if sent_id in axis_scores_dict:
                raw_logit = sent_logits[sent_id]
                axis_score = np.array(axis_scores_dict[sent_id])

                all_axis_raw_logits.append(raw_logit)
                all_axis_scores.append(axis_score)
                all_axis_sentence_ids.append((conv_idx, sent_id))

    if len(all_axis_scores) == 0:
        print("  ⚠️  No axis scores found, skipping axis correction")
    else:
        all_axis_raw_logits = np.array(all_axis_raw_logits)
        all_axis_scores = np.array(all_axis_scores)  # (n_sentences, 4)

        print(f"  Collected {len(all_axis_raw_logits)} axis sentence scores")

        # Compute correction for each axis
        print("\n[2/3] Computing axis corrections...")

        # Normalize raw logits
        axis_raw_normalized = (all_axis_raw_logits - all_axis_raw_logits.mean()) / all_axis_raw_logits.std()

        # Compute correction for each axis dimension separately
        axis_corrections = {}
        for axis_idx, axis in enumerate(['valence', 'arousal', 'dominance', 'approach_avoidance']):
            axis_scores = all_axis_scores[:, axis_idx]

            # Compute correlation
            corr = np.corrcoef(axis_scores, all_axis_raw_logits)[0, 1]

            # Compute optimal alpha to remove correlation
            alpha = corr * axis_scores.std() / axis_raw_normalized.std()

            axis_corrections[axis] = {
                'alpha': alpha,
                'correlation_before': corr
            }

            print(f"  {axis:20s}: corr={corr:6.3f}, alpha={alpha:6.3f}")

        # Also compute correction for mean axis score
        mean_axis_scores = all_axis_scores.mean(axis=1)
        axis_corr_mean = np.corrcoef(mean_axis_scores, all_axis_raw_logits)[0, 1]
        axis_alpha_mean = axis_corr_mean * mean_axis_scores.std() / axis_raw_normalized.std()

        print(f"\n  {'mean':20s}: corr={axis_corr_mean:6.3f}, alpha={axis_alpha_mean:6.3f}")

        # Apply corrections
        print("\n[3/3] Applying axis corrections...")

        # Use the mean alpha for all axes (simpler and more stable)
        axis_correction_vector = axis_alpha_mean * axis_raw_normalized

        n_axis_corrected = 0
        for idx, (conv_idx, sent_id) in enumerate(all_axis_sentence_ids):
            conv = data['conversations'][conv_idx]

            # Get original scores
            original_axis_scores = np.array(conv['probe_scores']['axis_lens_mean'][sent_id])

            # Apply correction (same correction to all axes)
            corrected_axis_scores = original_axis_scores - axis_correction_vector[idx]

            # Update in place
            conv['probe_scores']['axis_lens_mean'][sent_id] = corrected_axis_scores.tolist()
            n_axis_corrected += 1

        print(f"  ✓ Corrected {n_axis_corrected} axis sentences")

    # Verify
    print("\n" + "="*80)
    print("VERIFICATION")
    print("="*80)

    # Re-collect and check correlation
    new_emotion_scores = []
    new_raw_logits = []

    for conv_idx, conv in enumerate(data['conversations']):
        sent_logits = conv['sentence_mean_logits'].get('axis_lens_mean', {})
        emotion_scores_dict = conv['probe_scores'].get('logit_lens_mean', {})

        for sent_id in sent_logits.keys():
            if sent_id in emotion_scores_dict:
                new_raw_logits.append(sent_logits[sent_id])
                new_emotion_scores.append(np.mean(emotion_scores_dict[sent_id]))

    new_corr = np.corrcoef(new_emotion_scores, new_raw_logits)[0, 1]

    print(f"\nEMOTION SCORES:")
    print(f"  Correlation before: {corr_mean:.4f}")
    print(f"  Correlation after:  {new_corr:.4f}")

    if abs(new_corr) < 0.01:
        print("  ✓ SUCCESS: Emotion baseline correlation removed!")
    else:
        print(f"  ⚠ WARNING: Emotion correlation is {new_corr:.4f} (expected ~0)")

    # Verify axis scores if they exist
    if len(all_axis_scores) > 0:
        new_axis_scores = []
        new_axis_raw_logits = []

        for conv_idx, conv in enumerate(data['conversations']):
            sent_logits = conv['sentence_mean_logits'].get('axis_lens_mean', {})
            axis_scores_dict = conv['probe_scores'].get('axis_lens_mean', {})

            for sent_id in sent_logits.keys():
                if sent_id in axis_scores_dict:
                    new_axis_raw_logits.append(sent_logits[sent_id])
                    new_axis_scores.append(np.mean(axis_scores_dict[sent_id]))

        new_axis_corr = np.corrcoef(new_axis_scores, new_axis_raw_logits)[0, 1]

        print(f"\nAXIS SCORES:")
        print(f"  Correlation before: {axis_corr_mean:.4f}")
        print(f"  Correlation after:  {new_axis_corr:.4f}")

        if abs(new_axis_corr) < 0.01:
            print("  ✓ SUCCESS: Axis baseline correlation removed!")
        else:
            print(f"  ⚠ WARNING: Axis correlation is {new_axis_corr:.4f} (expected ~0)")

    # Save
    print("\n" + "="*80)
    print("SAVING")
    print("="*80)

    with open(output_path, 'wb') as f:
        pickle.dump(data, f)

    print(f"✓ Saved: {output_path}")
    print(f"  Size: {output_path.stat().st_size / 1e6:.1f} MB")

    print("\n" + "="*80)
    print("DONE!")
    print("="*80)

if __name__ == '__main__':
    input_path = Path('/workspace-vast/annas/git/research-tools/eval_dashboard/data/high_emotion_6plus_with_axes.pkl')
    apply_baseline_correction(input_path)
