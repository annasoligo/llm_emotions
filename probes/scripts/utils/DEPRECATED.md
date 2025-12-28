# DEPRECATION NOTICE

## ⚠️ This utilities module is DEPRECATED

**Date:** 2024-12-27

The utilities in this directory (`emo_lens_probe_utils.py`) are **deprecated** and should no longer be used for new code.

## Why Deprecated?

1. **No caching** - Reloads probes on every call
2. **Scattered functions** - No cohesive API
3. **Duplication** - Similar logic repeated across multiple scripts
4. **Limited functionality** - Doesn't support token-level analysis or orthogonal probes

## ✅ Use Instead: `probe_pipeline.py`

All functionality has been migrated to the new modular pipeline in:

```
probes/scripts/probe_pipeline.py
```

The new pipeline provides:
- ✅ **ProbeActivationExtractor** - Efficient activation extraction with caching
- ✅ **ProbeInference** - Probe loading and inference with automatic caching
- ✅ **ProbeAggregator** - Statistical operations (double-diff, bootstrap CI)
- ✅ **ProbeVisualizer** - Standardized visualizations

---

## Migration Guide

### Old Code (DEPRECATED):

```python
from probes.scripts.utils.emo_lens_probe_utils import (
    load_probe_and_cpca,
    apply_probe_pipeline,
    extract_activations_batch,
    compute_bootstrap_ci
)

# Load probe
probe_dict, cpca_components = load_probe_and_cpca(
    probe_path, cpca_path, layer, n_components
)

# Extract activations
activations = extract_activations_batch(
    model, tokenizer, prompts, layer,
    activation_strategy="assistant_token"
)

# Apply probe
logits = apply_probe_pipeline(
    activations, cpca_components, probe_dict, drop_neutral=True
)

# Bootstrap CI
ci = compute_bootstrap_ci(per_pair_effects, emotions, n_bootstrap, ci_percentile)
```

### New Code (✅ RECOMMENDED):

```python
from probes.scripts.probe_pipeline import (
    ProbeActivationExtractor,
    ProbeInference,
    ProbeAggregator
)

# Initialize once (with caching!)
extractor = ProbeActivationExtractor()
inference = ProbeInference(probe_dir, cpca_path)
aggregator = ProbeAggregator()

# Extract activations (same API)
activations = extractor.extract_aggregated(
    model, tokenizer, prompts, layer,
    strategy="assistant_token"
)

# Apply probe (automatic caching)
logits = inference.predict(
    activations, layer, n_components, drop_neutral=True
)

# Bootstrap CI (same logic)
ci = aggregator.bootstrap_ci(
    per_pair_effects, emotions, n_bootstrap, ci_percentile
)
```

---

## Key Improvements

### 1. Probe Caching

**Old:** Reloads probe from disk every time
```python
# Every call reloads from disk!
for layer in layers:
    probe, cpca = load_probe_and_cpca(path, cpca_path, layer, nc)
```

**New:** Loads once, caches automatically
```python
inference = ProbeInference(probe_dir, cpca_path)  # Load cPCA once
for layer in layers:
    logits = inference.predict(activations, layer, nc)  # Uses cache
```

### 2. Multi-Layer Efficiency

**Old:** Separate forward pass per layer
```python
activations_by_layer = {}
for layer in layers:
    acts = extract_activations_batch(model, tokenizer, prompts, layer)
    activations_by_layer[layer] = acts  # N forward passes!
```

**New:** Single forward pass for all layers
```python
extractor = ProbeActivationExtractor()
activations_by_layer = extractor.extract_batch_multilayer(
    model, tokenizer, prompts, layers  # 1 forward pass total!
)
```

### 3. Token-Level Analysis

**Old:** Not supported (need to write custom code)

**New:** Built-in method
```python
result = extractor.extract_token_level(
    model, tokenizer, prompt, layers,
    num_generate=100  # Extract every token!
)
```

### 4. Orthogonal Probes

**Old:** Not supported

**New:** Built-in support
```python
probe_data = inference.load_orthogonal_probe(
    layer, representation="raw", orthogonality_weight=1000.0
)

user_scores, asst_scores = inference.predict_orthogonal(
    activations,
    probe_data['final_user_probes'],
    probe_data['final_asst_probes'],
    emotions
)
```

### 5. Double-Diff Analysis

**Old:** Manual computation with verbose code
```python
D_ft = L_ft_ds - L_ft_bl
D_base = L_base_ds - L_base_bl
DD = D_ft - D_base
mean_effect = {emotions[j]: float(np.mean(DD[:, j])) for j in range(6)}
# ... more manual stats computation
```

**New:** One-liner with automatic bootstrap CI
```python
result = aggregator.compute_double_diff(
    ft_dataset=L_ft_ds,
    base_dataset=L_base_ds,
    ft_baseline=L_ft_bl,
    base_baseline=L_base_bl,
    emotions=emotions,
    n_bootstrap=1000
)
# Returns: mean_effect, bootstrap_ci, per_pair_effects, emotion_ranking
```

---

## Updated Entry Points

### For Interactive Analysis:

Use the new interactive notebooks:
- **Model diffing**: `probes/scripts/model_diff_analysis_interactive.py`
- **Token-level**: `probes/scripts/token_level_analysis_interactive.py` (migrated)

### For Batch Experiments:

Update your scripts to import from `probe_pipeline.py` instead of `emo_lens_probe_utils.py`.

---

## Timeline

- **Now (2024-12-27):** `emo_lens_probe_utils.py` marked as deprecated
- **Next 2 weeks:** Migrate existing scripts to use `probe_pipeline.py`
- **After migration:** `emo_lens_probe_utils.py` will be deleted

---

## Questions?

See:
- `probes/scripts/probe_pipeline.py` - New implementation
- `probes/scripts/model_diff_analysis_interactive.py` - Usage example
- `probes/PIPELINE_IMPLEMENTATION_SUMMARY.md` - Architecture documentation
