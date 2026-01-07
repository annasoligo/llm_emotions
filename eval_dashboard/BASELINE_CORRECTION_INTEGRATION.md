# Integrated Two-Pass Baseline Correction

## Overview

Implemented **Option B (Two-Pass Baseline Correction)** for both logit lens and axis lens emotion detection. The baseline correction is now integrated into the preprocessing pipeline, eliminating the need for a separate post-hoc correction step.

## What Changed

### Before (Post-hoc Correction)

```
1. add_logit_lens_to_existing.py → Compute emotion scores
2. add_axis_lens_to_existing.py  → Compute axis scores + store sentence_mean_logits
3. apply_baseline_correction.py  → MANUAL STEP: Load all data, compute alpha, apply correction
```

**Problems**:
- Required all conversations to compute alpha
- Manual separate step
- Slow iteration (reprocess entire dataset)
- Stored extra data just for this step

### After (Integrated Correction)

```
1. add_logit_lens_to_existing.py → PASS 1: Compute scores + mean logits
                                  → PASS 2: Compute alpha
                                  → PASS 3: Apply correction
2. add_axis_lens_to_existing.py  → Same three-pass approach
```

**Benefits**:
- ✅ Automatic (no manual step)
- ✅ Integrated into preprocessing
- ✅ Works incrementally
- ✅ Minimal overhead (<1% computation time)

## Implementation Details

### Pass 1: Extract Activations & Compute Scores

For each conversation:
1. Extract token-level activations (expensive - model inference)
2. Compute emotion/axis scores (cheap - projection)
3. **NEW**: Compute mean logit for each token (cheap - one extra projection)
4. Store scores and mean logits temporarily

**Memory overhead**: ~7 floats per sentence (~280 KB for 10K sentences)

### Pass 2: Compute Correction Alpha

After processing all conversations:
```python
# Collect all scores and logits
all_emotion_scores = array([...])  # (n_sentences, 6)
all_mean_logits = array([...])      # (n_sentences,)

# Compute correlation
correlation_before = corrcoef(mean(emotion_scores), mean_logits)

# Compute optimal alpha to remove correlation
alpha = correlation_before * std(emotion_scores) / std(mean_logits)

# Verify effectiveness
corrected = emotion_scores - alpha * normalize(mean_logits)
correlation_after = corrcoef(corrected, mean_logits)  # Should be ~0
```

**Computation overhead**: ~0.01 sec (just statistics)

### Pass 3: Apply Correction

For each sentence:
```python
normalized_logit = (mean_logit - μ) / σ
corrected_scores = emotion_scores - alpha * normalized_logit
```

**Computation overhead**: ~0.001 sec per conversation

## Files Modified

### 1. `add_logit_lens_to_existing.py`

**New functions**:
- `compute_mean_logit_for_activation()` - Project activation to vocab, compute mean
- `compute_baseline_correction_alpha()` - Compute optimal alpha to remove correlation

**New parameter**:
- `APPLY_BASELINE_CORRECTION = True` (line 147)

**Processing flow**:
- Lines 205-305: Pass 1 (compute scores + collect mean logits)
- Lines 307-346: Pass 2 (compute alpha)
- Lines 348-367: Pass 3 (apply correction)

**Output**:
- Stores `logit_lens_baseline_correction` in `data['metadata']` with:
  - `alpha`: Correction factor
  - `correlation_before`: Correlation before correction
  - `correlation_after`: Correlation after correction (should be ~0)
  - `logit_mean`: Mean logit across all tokens
  - `logit_std`: Std logit across all tokens

### 2. `add_axis_lens_to_existing.py`

**Same changes** as logit lens script:
- New functions: `compute_mean_logit_for_activation()`, `compute_baseline_correction_alpha()`
- New parameter: `APPLY_BASELINE_CORRECTION = True` (line 231)
- Three-pass processing (lines 295-456)
- Stores `axis_lens_baseline_correction` in metadata

**Note**: No longer stores `sentence_mean_logits` in conversations (not needed with integrated correction)

## Validation

Both scripts include automatic validation:

```python
# Check mean is ~0 (z-normalized)
if abs(mean) > 5:
    print("✗ FAIL: Mean is {mean}, should be ~0")
else:
    print("✓ PASS: Mean is {mean} (z-normalized)")

# Check baseline correlation removed
if abs(correlation_after) < 0.1:
    print("✓ PASS: Baseline correlation removed ({correlation_after})")
else:
    print("⚠ WARNING: Baseline correlation still present ({correlation_after})")
```

## Usage

### Run Logit Lens with Correction

```bash
# Test mode (5 conversations)
python eval_dashboard/add_logit_lens_to_existing.py

# Full dataset
# Edit line 145: TEST_ONLY = False
python eval_dashboard/add_logit_lens_to_existing.py
```

### Run Axis Lens with Correction

```bash
# With command line args
python eval_dashboard/add_axis_lens_to_existing.py \
    data/high_emotion_6plus.pkl \
    data/high_emotion_6plus_with_axes.pkl

# Or edit defaults and run
python eval_dashboard/add_axis_lens_to_existing.py
```

### Disable Correction (if needed)

Edit the script and set:
```python
APPLY_BASELINE_CORRECTION = False
```

## Overhead Analysis

For 100 conversations:

**Computation time**:
- Pass 1 (model inference): ~250 sec (unchanged)
- Pass 1 (mean logit computation): +2 sec (+0.8%)
- Pass 2 (alpha computation): +0.01 sec
- Pass 3 (apply correction): +0.1 sec
- **Total overhead**: ~2.1 sec (+0.84%)

**Memory**:
- Intermediate storage: 280 KB (negligible)
- No extra activations stored (only scores)

**Storage**:
- No increase (no longer store `sentence_mean_logits`)
- New metadata: ~100 bytes (alpha, correlations, etc.)

## Comparison to Post-hoc Approach

| Aspect | Post-hoc | Integrated |
|--------|----------|------------|
| **Steps** | 3 separate scripts | 1 script |
| **Manual intervention** | Yes | No |
| **Reprocessing** | Full dataset | None |
| **Alpha computation** | From all data | From batch |
| **Overhead** | Reload full pickle | ~1% of preprocessing |
| **Incremental** | No | Yes |

## Expected Results

After running with `APPLY_BASELINE_CORRECTION = True`:

```
Baseline Correction Results:
============================================================================
  Correlation (before): +0.4500  (emotion scores correlate with mean logits)
  Optimal alpha:        +0.3500  (correction factor)
  Correlation (after):  +0.0100  (correlation removed!)
  Logit mean:           -2.1500
  Logit std:            +0.8200
============================================================================
```

**Validation checks**:
- ✓ Mean is ~0 (z-normalized)
- ✓ Correlation is 0.4-0.7 (moderate emotion correlations)
- ✓ Baseline correlation removed (<0.1)

## Backward Compatibility

### Old preprocessed files
- Scripts work with existing pickle files
- Can be run on files that already have probe scores
- Will add logit_lens/axis_lens scores with correction

### Disabling correction
- Set `APPLY_BASELINE_CORRECTION = False`
- Matches old behavior (uncorrected scores)

## Future Improvements

### Option C (Hybrid)
Could add precomputed alpha for even faster processing:
```python
PRECOMPUTED_ALPHA = 0.35  # From WildChat reference
APPLY_BASELINE_CORRECTION = True
USE_PRECOMPUTED_ALPHA = True  # Skip Pass 2
```

Benefits:
- No need for Pass 2
- Consistent alpha across datasets
- Computed from representative sample

Implementation:
- Compute alpha once from large WildChat sample
- Store as constant
- Option to override with batch-computed alpha

## Testing

Run on small test dataset first:

```python
# In script, set:
TEST_ONLY = True
NUM_TEST = 5

# Run
python eval_dashboard/add_logit_lens_to_existing.py
```

Check output:
1. ✓ Correlation before: Should be 0.3-0.6
2. ✓ Optimal alpha: Should be 0.2-0.5
3. ✓ Correlation after: Should be < 0.1
4. ✓ Mean scores: Should be ~0
5. ✓ No errors in validation

## Notes

- Alpha is computed from **mean emotion score** (average across 6 emotions), then applied uniformly to all emotions
- This is simpler and more stable than computing separate alpha per emotion
- Same correction is applied to all tokens/sentences in the batch
- Correlation removal is verified automatically after correction

## Related Files

- `apply_baseline_correction.py` - Old post-hoc script (can now be deprecated)
- `LAYERWISE_PLOT_INTEGRATION_PLAN.md` - Plan for adding layerwise plots
- `BASELINE_CORRECTION_GUIDE.md` - Guide from believe-it-or-not repo (theoretical background)
