# Unified Preprocessing Pipeline Plan

## Executive Summary

This plan addresses two issues:
1. **No single-pass pipeline** - Model loaded 3x instead of once
2. **Missing per-layer probe scores** - Layerwise plots work for logit lens but not probes

**Critical concern**: Normalization must be handled correctly for per-layer scores.

---

## Current Normalization Architecture

### How Aggregated Probe Scores Are Normalized

```
data_preprocessing.py (lines 123-168, 332-341):

1. Extract activations for each token at layers 20-40
   └─ activations_by_token[token_pos][layer] = activation

2. Apply probe to each layer's activation
   └─ scores_by_token[token_pos][layer] = probe(activation)
   └─ Returns: softmax probabilities [6] (0-1 range)

3. AGGREGATE across layers (BEFORE normalization)
   └─ aggregated_scores[token_pos] = np.mean([layer_scores for layers 20-40])

4. Z-SCORE NORMALIZE using LAYER-AVERAGED baseline
   └─ baseline = mean/std computed from WildChat, averaged across layers 20-40
   └─ normalized = (aggregated_score - baseline_mean) / baseline_std

5. Aggregate tokens → sentences
```

**Key point**: Baseline is computed AFTER averaging across layers, so it reflects the distribution of layer-averaged probe scores.

### How Logit Lens Per-Layer Scores Are Normalized

```
add_logit_lens_to_existing.py (lines 361-381, 495-507):

1. Extract activations for ALL layers

2. For EACH layer separately:
   └─ Project activation to vocabulary space
   └─ Get logits for emotion tokens
   └─ Z-score normalize using PER-LAYER baseline statistics
       └─ ref_stats['layers_data'][str(layer)]['statistics'][token_id]
   └─ Store per-layer score

3. Apply GLOBAL baseline correction (two-pass):
   └─ Compute correlation between emotion scores and mean logits
   └─ Remove correlation: corrected = score - alpha * normalized_logit
   └─ This is applied to BOTH aggregated AND per-layer scores
```

**Key difference**: Logit lens uses PER-LAYER baselines, probes use LAYER-AVERAGED baselines.

---

## The Normalization Decision

### Option A: Layer-Averaged Baseline for Per-Layer Scores (RECOMMENDED)

```
For each layer L:
  per_layer_score[L] = (raw_probe_score[L] - layer_avg_baseline_mean) / layer_avg_baseline_std
```

**Pros**:
- **Consistent** with aggregated scores (same baseline reference)
- **Interpretable**: Shows how each layer deviates from the population mean
- **Cross-layer comparable**: All layers on same scale
- **Simple**: No new baselines needed

**Cons**:
- Ignores layer-specific baseline characteristics
- Early layers might have systematically different distributions

**Interpretation**: "How does layer L compare to the average across all layers?"

### Option B: Per-Layer Baselines

```
For each layer L:
  per_layer_score[L] = (raw_probe_score[L] - baseline_mean[L]) / baseline_std[L]
```

**Pros**:
- More accurate per-layer normalization

**Cons**:
- **Requires per-layer baseline statistics** (may not exist for all probes)
- **Cross-layer comparison confounded** by different baselines
- **Inconsistent** with aggregated scores

### Option C: No Normalization (Current Logit Lens Approach Before Correction)

Store raw softmax probabilities.

**Cons**:
- **Scale mismatch**: Aggregated scores are z-scores (σ units), per-layer are probabilities (0-1)
- **Misleading plots**: Y-axis says "σ" but data is in different units
- **Not recommended**

---

## Recommendation: Option A with Clear Documentation

Use the **same layer-averaged baseline** for both:
- Aggregated scores (current behavior)
- Per-layer scores (new behavior)

This ensures:
1. **Consistency**: All scores on same scale
2. **Interpretability**: Cross-layer plots show relative differences
3. **No new dependencies**: Uses existing baseline infrastructure

---

## Implementation Plan

### Phase 1: Modify `data_preprocessing.py` to Store Per-Layer Scores

**Changes to `apply_probes_to_activations()` (lines 123-168)**:

```python
def apply_probes_to_activations(
    activations_by_token: Dict[int, Dict[int, np.ndarray]],
    probe_experiment: TokenLevelExperiment,
    layers: List[int],
    store_per_layer: bool = True  # NEW PARAMETER
) -> Tuple[Dict[int, np.ndarray], Optional[Dict[int, Dict[int, np.ndarray]]]]:
    """
    Returns:
        Tuple of:
        - aggregated_scores: {token_pos -> scores[6]}  (layer-averaged)
        - per_layer_scores: {token_pos -> {layer -> scores[6]}}  (optional)
    """
    scores_by_token = probe_experiment._apply_probes(...)

    aggregated_scores = {}
    per_layer_scores = {} if store_per_layer else None

    for token_pos in scores_by_token:
        # ... existing aggregation logic ...

        # NEW: Also store per-layer scores
        if store_per_layer:
            per_layer_scores[token_pos] = {}
            for layer in layers:
                per_layer_scores[token_pos][layer] = scores_by_token[token_pos][layer]

    return aggregated_scores, per_layer_scores
```

**Changes to normalization (lines 332-350)**:

```python
# Normalize BOTH aggregated and per-layer scores with SAME baseline
probe_mean = probe_baselines[probe_key]['mean']
probe_std = probe_baselines[probe_key]['std']

# Normalize aggregated
for token_pos in token_scores:
    token_scores[token_pos] = normalize_probe_scores_zscore(
        token_scores[token_pos], probe_mean, probe_std
    )

# NEW: Normalize per-layer with SAME baseline
if per_layer_scores:
    for token_pos in per_layer_scores:
        for layer in per_layer_scores[token_pos]:
            per_layer_scores[token_pos][layer] = normalize_probe_scores_zscore(
                per_layer_scores[token_pos][layer], probe_mean, probe_std
            )
```

**Changes to storage**:

```python
# Aggregate per-layer scores to sentence level
if per_layer_scores:
    per_layer_key = f"{probe_key}_by_layer"
    sentence_per_layer_scores = aggregate_per_layer_to_sentences(
        sentences=sentences,
        per_layer_scores=per_layer_scores,
        layers=MODEL_CONFIG['layers']
    )
    conv[per_layer_key] = sentence_per_layer_scores
```

### Phase 2: Unified Single-Pass Pipeline

Create `preprocess_unified.py` that:

1. Loads model ONCE
2. For each conversation:
   - Extract activations (all layers needed by any probe type)
   - Apply ALL probe types
   - Apply logit lens
   - Apply axis lens
3. Normalize everything with consistent baselines
4. Save single pickle

**Layers to extract**: Union of all layer requirements
- Probes: 20-40
- Logit lens: 0-61 (or model max)
- Result: Extract all layers 0-61, use subsets as needed

### Phase 3: Update `layerwise_plotting.py`

Currently expects dict format from logit lens. Need to handle both:

```python
# Line 120-124: Update to handle both array and dict formats
layer_scores = sent_layers[layer]

# Handle different formats
if isinstance(layer_scores, dict):
    # Logit lens format: {emotion: score}
    for emotion_idx, emotion in enumerate(EMOTIONS):
        scores_by_emotion[emotion][layer].append(layer_scores[emotion])
elif isinstance(layer_scores, (list, np.ndarray)):
    # Probe format: [score1, score2, ...]
    for emotion_idx, emotion in enumerate(EMOTIONS):
        scores_by_emotion[emotion][layer].append(layer_scores[emotion_idx])
```

---

## Data Format Specification

### Aggregated Scores (unchanged)

```python
conv['probe_scores'][probe_key][sent_id] = np.ndarray([6])  # z-scores
```

### Per-Layer Scores (new)

```python
conv[f'{probe_key}_by_layer'][sent_id][layer] = np.ndarray([6])  # z-scores

# Example:
conv['orthogonal_raw_by_layer'][15][25] = array([0.5, -0.2, 1.1, ...])
#                               ^   ^
#                         sent_id  layer
```

### Orthogonal Probes (special case)

```python
# Aggregated (current)
conv['probe_scores']['orthogonal_raw'][sent_id] = {
    'user': np.ndarray([6]),
    'assistant': np.ndarray([6])
}

# Per-layer (new)
conv['orthogonal_raw_by_layer'][sent_id][layer] = {
    'user': np.ndarray([6]),
    'assistant': np.ndarray([6])
}
```

---

## Normalization Summary Table

| Probe Type | Aggregated Normalization | Per-Layer Normalization |
|------------|-------------------------|------------------------|
| orthogonal_raw | Z-score (layer-avg baseline) | Z-score (SAME layer-avg baseline) |
| text_raw | Z-score (layer-avg baseline) | Z-score (SAME layer-avg baseline) |
| centroid_k10 | Z-score (layer-avg baseline) | Z-score (SAME layer-avg baseline) |
| logit_lens | Per-layer z-score + baseline correction | Per-layer z-score + baseline correction |

**Key principle**: For probes, use the SAME baseline for both aggregated and per-layer scores.

---

## Verification Checklist

After implementation, verify:

1. [ ] Aggregated scores unchanged (regression test)
2. [ ] Per-layer scores have same mean/std characteristics as aggregated
3. [ ] Layerwise plots render correctly for both probes and logit lens
4. [ ] Cross-layer comparison shows sensible patterns (not flat lines)
5. [ ] Y-axis label "σ" is accurate (all scores are z-scores)

---

## Questions for Review

1. **Layer range**: Should per-layer scores cover layers 20-40 (current probe range) or expand to 0-61 (full model)?

2. **Orthogonal probes**: Store both user and assistant per-layer, or just one?

3. **Memory**: Per-layer storage increases pickle size ~20x. Acceptable?

4. **Backward compatibility**: Keep old pickle format, or require re-preprocessing?

---

## Files to Modify

1. `data_preprocessing.py` - Add per-layer storage and normalization
2. `preprocess_unified.py` (new) - Single-pass pipeline
3. `layerwise_plotting.py` - Handle both probe and logit lens formats
4. `probe_configs.py` - Add per-layer configuration options
5. `sentence_aggregator.py` - Add per-layer aggregation function

---

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| Normalization inconsistency | Use SAME baseline for aggregated and per-layer |
| Breaking existing analysis | Keep aggregated scores unchanged |
| Memory explosion | Make per-layer storage optional |
| Plot misinterpretation | Document units clearly in plot labels |
