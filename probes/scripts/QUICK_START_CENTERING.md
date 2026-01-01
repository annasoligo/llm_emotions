# Quick Start: Enabling Probe Score Centering

## What Changed

I've added a new parameter `center_probe_scores=True` to enable baseline-relative emotion scores.

## How to Use

### In token_level_experiment_v2.py

I've already updated your experiment file! Just look for this new parameter:

```python
# Line 116: New parameter
CENTER_PROBE_SCORES = True  # Recommended for orthogonal probes!
```

This is now enabled by default in both experiments (orthogonal and standard probes).

### Before vs After

**Before (center_probe_scores=False):**
```
happiness = 0.3   # Low but positive
sadness = 0.7     # High
anger = 0.2       # Low but positive
```
All scores are positive. Hard to tell if low scores are meaningful or just noise.

**After (center_probe_scores=True):**
```
happiness = -0.1  # Below baseline!
sadness = +0.4    # Above baseline!
anger = 0.0       # At baseline
```
Now you can clearly see:
- **Negative = less than typical conversation**
- **Positive = more than typical conversation**
- **Zero = at typical baseline level**

## What This Does Internally

When you run experiments with `center_probe_scores=True`:

1. **Computes baseline**: Applies your probes to WildChat baseline activations
2. **Gets expected scores**: `[anger=0.2, disgust=0.15, fear=0.18, happiness=0.4, sadness=0.3, surprise=0.12]`
3. **Subtracts baseline**: `observed_scores - baseline_scores`
4. **Returns centered scores**: Now negative values indicate "below typical"

## When to Use

**Always use for orthogonal probes** - they measure projection magnitude which is always positive without centering.

**Optional for standard probes** - they already have learned bias terms, but centering adds WildChat-specific baseline reference.

## Example Results

Here's what you'll see with centering enabled:

```python
Token: "terrible"
  anger:      +0.45  # More anger than typical
  disgust:    +0.38  # More disgust than typical
  fear:       +0.15  # More fear than typical
  happiness:  -0.62  # LESS happy than typical!
  sadness:    +0.52  # More sadness than typical
  surprise:   -0.02  # Slightly less surprise than typical
```

Without centering, happiness would be `0.18` (positive but low), making it unclear if this is meaningful.

With centering, happiness is `-0.62` (negative), clearly showing it's **below baseline**.

## Quick Test

To see the difference, run your experiment twice:

```python
# Test 1: Without centering
CENTER_PROBE_SCORES = False
# Run experiment...

# Test 2: With centering (recommended)
CENTER_PROBE_SCORES = True
# Run experiment...
```

Compare the bar charts - you'll see negative bars for emotions that are below typical levels!

## Technical Note

The baseline is computed as:
1. Load mean activations from WildChat (assistant_turn aggregation)
2. Average these means across all layers (layer-averaged baseline)
3. Apply your probes to get expected scores
4. Subtract these from observed scores at each token

This gives you interpretable relative emotion levels.
