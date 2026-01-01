# Probe Score Baseline Centering

## Problem Statement

When using orthogonal emotion probes, all scores are positive because they measure the **magnitude of projection** onto emotion direction vectors. This makes it difficult to interpret whether an emotion is "above baseline" or "below baseline" - you can only tell if it's "more present" or "less present."

For example:
- `happiness = 0.8`: Strong alignment with happiness direction
- `happiness = 0.2`: Weak alignment with happiness direction

But what if you want to know: **Is this MORE happy than a typical conversation, or LESS happy?**

## Solution: Baseline Centering

We now support **probe score baseline centering**, which:

1. Computes expected probe scores on WildChat baseline activations (typical conversations)
2. Subtracts these baseline scores from observed scores
3. Gives you **relative emotion levels**: positive = above baseline, negative = below baseline

## Usage

### Enable in TokenLevelExperiment

```python
from probes.scripts.token_level_helpers import TokenLevelExperiment

exp = TokenLevelExperiment(
    model=model,
    tokenizer=tokenizer,
    probe_type="orthogonal",
    probe_dir=Path("outputs/probes/emotion_probes/conversation_based/"),
    cpca_path=Path("outputs/cpca/conversation_based/global/google/gemma-3-27b-it_cpca.npz"),
    use_wildchat_normalization=True,  # Step 1: Normalize activations
    center_probe_scores=True,         # Step 2: Center probe outputs (NEW!)
    wildchat_aggregation="assistant_turn"
)

results = exp.run_experiment(
    prompt="Your prompt here",
    layers=list(range(30, 60)),
    num_generated_tokens=50
)

# Now scores can be negative!
# Positive = above baseline
# Negative = below baseline
```

### Standalone Usage

You can also compute baselines manually:

```python
from probes.scripts.wildchat_baseline_loader import WildChatBaselineLoader
from probes.scripts.probe_pipeline import ProbeInference

# Initialize
loader = WildChatBaselineLoader(aggregation_type="assistant_turn")
inference = ProbeInference(probe_dir=probe_dir, cpca_path=cpca_path)

# Compute baseline probe scores
baselines = loader.compute_probe_score_baselines(
    probe_inference=inference,
    layers=list(range(30, 60)),
    probe_type="orthogonal",
    aggregation="mean",  # Use layer-averaged baseline
    orthogonality_weight=1000.0,
    orthogonal_representation="raw",
    emotions=['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
)

# baselines is a dict: {-1: array([baseline_anger, baseline_disgust, ...])}
baseline_vec = baselines[-1]  # -1 indicates layer-averaged

# Subtract from observed scores
centered_scores = observed_scores - baseline_vec
```

## Interpretation

### Without Centering (Default)
```python
center_probe_scores=False  # Default
```

**All scores are positive:**
- `happiness = 0.8`: Strong presence of happiness
- `happiness = 0.2`: Weak presence of happiness
- `anger = 0.5`: Moderate presence of anger

**Interpretation**: Absolute projection magnitude onto emotion directions.

### With Centering (New!)
```python
center_probe_scores=True
```

**Scores can be positive or negative:**
- `happiness = +0.3`: More happy than typical conversation
- `happiness = -0.2`: Less happy than typical conversation
- `anger = +0.5`: More anger than typical conversation
- `anger = -0.1`: Less anger than typical conversation

**Interpretation**: Relative deviation from typical WildChat conversations.

## Why This Matters

### Example: Analyzing a Sad Story

**Without centering:**
```
happiness = 0.3   # Low but positive
sadness = 0.7     # High
anger = 0.2       # Low but positive
```

**Interpretation issue**: Is the low happiness score meaningful, or is it just "background noise"?

**With centering (assume baseline is happiness=0.4, sadness=0.3, anger=0.2):**
```
happiness = -0.1   # Below baseline (notably less happy)
sadness = +0.4     # Above baseline (notably more sad)
anger = 0.0        # At baseline (typical level)
```

**Clear interpretation**: The story makes the model **less happy** and **more sad** than typical, while anger remains at baseline.

## Technical Details

### How Baselines Are Computed

1. **Load WildChat baseline activations** from pre-computed statistics (mean activation across WildChat dataset)
2. **Apply probes to baseline activations** to get expected scores
3. **Average across layers** (recommended) or use per-layer baselines
4. **Subtract from observed scores** at each token position

### Aggregation Options

```python
# Option 1: Layer-averaged baseline (recommended)
aggregation="mean"
# Computes single baseline vector by:
# 1. Averaging baseline activations across layers
# 2. Applying probes to get baseline scores
# Result: One baseline for all layers

# Option 2: Per-layer baselines
aggregation="per_layer"
# Computes separate baseline for each layer
# More complex, harder to interpret across layers
```

### Works with All Probe Types

- **Orthogonal probes**: Most impactful (transforms always-positive to centered scores)
- **Standard probes**: Less impactful (already have bias term for centering, but this adds WildChat-specific baseline)
- **Linear probes**: Similar to orthogonal (no bias term, so centering helps interpretation)

## Visualization

The existing visualization functions already support negative values:

```python
from probes.scripts.model_diff_viz import plot_sentence_level_bar_charts

# Works with both centered and non-centered scores
plot_sentence_level_bar_charts(
    scores_dict=aggregated_scores,
    emotions=EMOTIONS,
    token_strings=token_strings,
    token_ids=token_ids,
    prompt_start_idx=20,
    title="Sentence-Level Emotions (Baseline-Centered)"
)
```

Bar charts automatically:
- Draw a black line at y=0
- Position value labels above bars (positive) or below bars (negative)
- Use same color scheme for all emotions

## Best Practices

1. **Always use with WildChat normalization**:
   ```python
   use_wildchat_normalization=True  # Normalize activations
   center_probe_scores=True         # Center probe scores
   ```

2. **Use layer-averaged baselines** for token-level analysis (enables cross-layer comparison)

3. **Standard probes have less dramatic change** since they already have learned bias terms

4. **Interpret relative to WildChat**: Baseline represents "typical conversational emotion," not "emotionally neutral"

## Example Results

Here's what you might see with baseline centering enabled:

```
Token: "That's"
  anger:      -0.12  (below baseline)
  disgust:    -0.08  (below baseline)
  fear:       -0.05  (slightly below baseline)
  happiness:  +0.25  (above baseline)
  sadness:    -0.15  (below baseline)
  surprise:   +0.03  (slightly above baseline)

Token: "terrible"
  anger:      +0.45  (notably above baseline)
  disgust:    +0.38  (notably above baseline)
  fear:       +0.15  (above baseline)
  happiness:  -0.62  (notably below baseline)
  sadness:    +0.52  (notably above baseline)
  surprise:   -0.02  (slightly below baseline)
```

This clearly shows "terrible" triggers negative emotions above baseline and positive emotions below baseline.

## Files Modified

- [`wildchat_baseline_loader.py`](wildchat_baseline_loader.py): Added `compute_probe_score_baselines()` method
- [`token_level_helpers.py`](token_level_helpers.py): Added `center_probe_scores` parameter to `TokenLevelExperiment`
- [`TOKEN_LEVEL_README.md`](TOKEN_LEVEL_README.md): Updated documentation with centering examples

## Related Documentation

- [TOKEN_LEVEL_README.md](TOKEN_LEVEL_README.md): Full token-level analysis documentation
- [WildChat Baselines](../data/baselines/wildchat/): Pre-computed baseline statistics
