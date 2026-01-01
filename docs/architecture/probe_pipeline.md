# Probe Application Audit Report

**Date:** 2026-01-01
**Scope:** Model diffing analysis, token-level experiments, and evaluation dashboard

## Executive Summary

This audit examines how emotion probes are applied across three systems:
1. **Model Diffing** (`model_diff_analysis_v2.py` + `model_diff_helpers.py`)
2. **Token-Level Analysis** (`token_level_experiment_v2.py` + `token_level_helpers.py`)
3. **Evaluation Dashboard** (`eval_dashboard/app.py` + `probe_configs.py`)

### Key Findings

✅ **Strengths:**
- Unified probe inference pipeline shared across systems
- Consistent nested dict format for orthogonal/centroid conversation probes
- Flexible normalization options (activation-level and probe-score-level)
- Good caching support for activation reuse

⚠️ **Issues Identified:**
- **Critical**: Inconsistent normalization defaults between scripts
- **High**: Probe score normalization applied AFTER probe inference, not stored in preprocessed data
- **Medium**: Hardcoded paths throughout all scripts
- **Medium**: Multiple normalization flags with unclear precedence
- **Low**: Documentation gaps on normalization semantics

---

## 1. Probe Loading & Application

### 1.1 Supported Probe Types

All three systems support the following probe types via `ProbeInference`:

| Probe Type | Description | Output Format | User/Asst Split |
|------------|-------------|---------------|-----------------|
| `orthogonal` | Orthogonal conversation probes | Nested dict | ✅ Yes |
| `linear` | Linear probes with optional cPCA | Flat array | ❌ No |
| `standard` | Custom pattern probes (multiseed) | Flat array | ❌ No |
| `centroid` | K-averaged centroid probes | Depends on format | Depends on format |

**Format Details:**
- **Flat array**: `np.ndarray` shape `[n_emotions]` (6 emotions)
- **Nested dict**: `{'user': np.ndarray[6], 'assistant': np.ndarray[6]}`

### 1.2 Probe Inference Pipeline

Both `DoubleDiffExperiment` and `TokenLevelExperiment` use the same `ProbeInference` class:

```python
# From probe_pipeline.py (inferred usage)
inference = ProbeInference(
    probe_dir=probe_dir,
    cpca_path=cpca_path,
    device='cuda'
)
```

**Methods used:**
- `predict()` - Linear probes
- `predict_orthogonal()` - Orthogonal probes
- `predict_centroid()` - Centroid probes
- `load_orthogonal_probe()` - Load orthogonal probe weights
- `load_centroid_probe()` - Load centroid probe weights

### 1.3 Probe Path Construction

**Orthogonal Probes:**
```python
probe_dir / f"layer{layer}_ortho_w{orthogonality_weight}_{representation}.pkl"
# Example: layer30_ortho_w1000.0_raw.pkl
```

**Standard Probes:**
```python
probe_dir / probe_pattern.format(layer=layer)
# Example pattern: "probe_layer{layer}_nc0_seed0.pkl"
# Result: probe_layer30_nc0_seed0.pkl
```

**Centroid Probes:**
```python
probe_dir / f"layer{layer}_k{k_value}_w{orthogonality_weight}_{format}.pkl"
# Example: layer30_k50_w100000.0_conversation.pkl
```

---

## 2. Normalization Architecture

### 2.1 Two-Stage Normalization

The system supports **two independent normalization stages**:

```
Raw Activations
      ↓
[Stage 1: Activation Normalization] ← WildChat baseline statistics (optional)
      ↓
Normalized Activations
      ↓
[Apply Probes]
      ↓
Raw Probe Scores
      ↓
[Stage 2: Probe Score Normalization] ← WildChat probe score statistics (optional)
      ↓
Final Scores
```

### 2.2 Stage 1: Activation Normalization

**Purpose:** Normalize activations BEFORE applying probes

**Controlled by:**
- `use_wildchat_normalization` (boolean)
- `baseline_dir` (Path to WildChat baseline statistics)

**Implementation:**
```python
# model_diff_helpers.py:266-294
if self.use_wildchat_normalization:
    baseline_loader = WildChatBaselineLoader(
        aggregation_type=wildchat_aggregation,
        baseline_dir=self.baseline_dir
    )
    normalized = {}
    for name, acts in activations.items():
        normalized[name] = baseline_loader.normalize_batch_multilayer(acts)
```

**Aggregation mapping:**
```python
activation_strategy → wildchat_aggregation
'assistant_token'      → 'first_assistant_token'
'last_user_token'      → 'last_user_token'
'between_turns_avg'    → 'between_turns'
'generated_tokens_avg' → 'assistant_turn'
```

### 2.3 Stage 2: Probe Score Normalization

**Purpose:** Normalize probe outputs AFTER inference

**Controlled by:**
- `normalize_probe_scores` (boolean) - Z-score normalization (subtract mean, divide by std)
- `center_probe_scores` (boolean) - Centering only (subtract mean)
- If both are True, `normalize_probe_scores` takes precedence

**Implementation:**
```python
# model_diff_helpers.py:470-594
if self.normalize_probe_scores:
    # Compute baseline statistics (mean + std)
    baseline_stats = baseline_loader.compute_probe_score_baselines(...)
    baseline_mean = baseline_stats['mean'][-1]
    baseline_std = baseline_stats['std'][-1]

    # Apply z-score normalization
    normalized[i] = normalize_probe_scores_zscore(
        scores[i], baseline_mean, baseline_std
    )
elif self.center_probe_scores:
    # Compute baseline mean only
    baseline_scores = baseline_loader.compute_probe_score_baselines(...)
    baseline_mean = baseline_scores[-1]

    # Apply centering
    normalized[i] = normalize_probe_scores_center(
        scores[i], baseline_mean
    )
```

**Normalization functions:**
```python
# From probe_pipeline.py (inferred)
def normalize_probe_scores_zscore(scores, mean, std):
    """Z-score: (x - mean) / std"""
    if isinstance(scores, dict):  # Handle nested dict format
        return {
            'user': (scores['user'] - mean) / std,
            'assistant': (scores['assistant'] - mean) / std
        }
    else:
        return (scores - mean) / std

def normalize_probe_scores_center(scores, mean):
    """Centering: x - mean"""
    if isinstance(scores, dict):
        return {
            'user': scores['user'] - mean,
            'assistant': scores['assistant'] - mean
        }
    else:
        return scores - mean
```

---

## 3. System-Specific Implementation

### 3.1 Model Diffing Analysis (`model_diff_analysis_v2.py`)

**Configuration:**
```python
USE_BASELINE_NORMALIZATION = False      # Stage 1: Activation normalization
NORMALIZE_PROBE_SCORES = True           # Stage 2: Z-score normalization
```

**Experiment 1: Orthogonal Probes**
```python
exp1 = DoubleDiffExperiment(
    base_model=base_model,
    ft_model=ft_model,
    tokenizer=tokenizer,
    probe_type="orthogonal",
    probe_dir=Path(".../conversation_based/"),
    cpca_path=Path(".../gemma-3-27b-it_cpca.npz"),
    orthogonality_weight=1000.0,
    orthogonal_representation="raw",
    use_wildchat_normalization=USE_BASELINE_NORMALIZATION,  # False
    normalize_probe_scores=NORMALIZE_PROBE_SCORES,          # True
    baseline_dir=BASELINE_DIR
)
```

**Experiment 2: Non-Orthogonal (Standard) Probes**
```python
exp2 = DoubleDiffExperiment(
    probe_type="standard",
    probe_dir=Path(".../text_based/multiseed/"),
    probe_pattern="probe_layer{layer}_nc0_seed0.pkl",
    cpca_path=None,  # No cPCA needed
    use_wildchat_normalization=USE_BASELINE_NORMALIZATION,  # False
    normalize_probe_scores=NORMALIZE_PROBE_SCORES,          # True
    baseline_dir=BASELINE_DIR
)
```

**Activation Caching:**
```python
# Exp2 reuses activations from exp1 (efficient!)
results_text_raw = exp2.run_experiment(
    ...,
    cached_activations=results_ortho_conv['activations']
)
```

### 3.2 Token-Level Analysis (`token_level_experiment_v2.py`)

**Configuration:**
```python
USE_BASELINE_NORMALIZATION = False      # Stage 1: Activation normalization
NORMALIZE_PROBE_SCORES = True           # Stage 2: Z-score normalization
CENTER_PROBE_SCORES = False             # Superseded by NORMALIZE_PROBE_SCORES
BASELINE_AGGREGATION = "all_tokens"
```

**Experiment 1: Orthogonal Probes**
```python
exp1 = TokenLevelExperiment(
    model=model,
    tokenizer=tokenizer,
    probe_type="orthogonal",
    probe_dir=Path(".../conversation_based/"),
    cpca_path=Path(".../gemma-3-27b-it_cpca.npz"),
    orthogonality_weight=1000.0,
    orthogonal_representation="raw",
    use_wildchat_normalization=USE_BASELINE_NORMALIZATION,  # False
    normalize_probe_scores=NORMALIZE_PROBE_SCORES,          # True
    wildchat_aggregation=BASELINE_AGGREGATION,
    baseline_dir=BASELINE_DIR,
    center_probe_scores=CENTER_PROBE_SCORES  # False
)
```

**Experiment 2: Standard (Linear) Probes**
```python
exp2 = TokenLevelExperiment(
    probe_type="linear",  # Note: different from model_diff "standard"
    probe_dir=Path(".../text_based/multiseed/"),
    probe_pattern="probe_layer{layer}_nc0_seed0.pkl",
    cpca_path=Path(".../gemma-3-27b-it_cpca.npz"),
    use_wildchat_normalization=USE_BASELINE_NORMALIZATION,  # False
    normalize_probe_scores=NORMALIZE_PROBE_SCORES,          # True
    n_components=10
)
```

**Experiment 3: Centroid Probes**
```python
exp3 = TokenLevelExperiment(
    probe_type="centroid",
    probe_dir=Path(".../conversation/"),
    k_value=10,
    orthogonality_weight=100000.0,
    centroid_probe_format="conversation",
    use_wildchat_normalization=False,
    normalize_probe_scores=NORMALIZE_PROBE_SCORES,  # True
    center_probe_scores=False
)
```

**Key Difference from Model Diffing:**
- Extracts token-by-token activations during generation
- Returns `scores_by_token` dict: `{token_pos: {layer: scores}}`
- Caches raw activations before normalization for reuse

### 3.3 Evaluation Dashboard (`eval_dashboard/app.py`)

**Architecture:**
- Dashboard does NOT apply probes directly
- Expects preprocessed conversation data with probe scores already computed
- Loads from pickle file: `preprocessed_conversations.pkl`

**Expected Data Structure:**
```python
{
    'conversations': [
        {
            'sample_id': int,
            'sentences': [
                {
                    'sentence_id': int,
                    'text': str,
                    'turn_role': 'user' | 'assistant',
                    'turn_index': int,
                    'start_token': int,
                    'end_token': int
                },
                ...
            ],
            'probe_scores': {
                'probe_key': {
                    sentence_id: scores,  # np.ndarray or dict
                    ...
                }
            },
            'metadata': {...}
        },
        ...
    ]
}
```

**Probe Score Format Detection:**
```python
# app.py:594-609
sample_score = list(sentence_scores.values())[0]
is_orthogonal = isinstance(sample_score, dict) and 'user' in sample_score

if is_orthogonal:
    # Use orthogonal plot with user/assistant subplots
    fig = create_orthogonal_trajectory_plot(...)
else:
    # Use standard plot
    fig = create_trajectory_plot(...)
```

**Smoothing/Re-aggregation:**
```python
# app.py:25-89
def smooth_sentence_scores(sentences, sentence_scores, window_size):
    """Re-aggregate sentence scores with different window size"""
    if window_size == 20:
        return sentence_scores  # Original chunking

    # Expand to token-level, then re-chunk
    token_scores = {}
    for sent in sentences:
        for tok in range(sent['start_token'], sent['end_token']):
            token_scores[tok] = sentence_scores[sent['sentence_id']]

    # Re-chunk with new window
    smoothed = {}
    for window_start in range(0, max_token + 1, window_size):
        window_scores = [token_scores[tok] for tok in range(window_start, window_end)]
        smoothed[sent_id] = np.mean(window_scores, axis=0)
```

**Probe Configuration:**
```python
# probe_configs.py defines 6 probe types
PROBE_CONFIGS = {
    'orthogonal_raw': {...},
    'orthogonal_cpca_top20': {...},
    'text_raw': {...},
    'text_cpca': {...},
    'centroid_k10': {...},
    'centroid_k50': {...}
}

# Baseline settings
BASELINE_CONFIG = {
    'normalize_probe_scores': True,      # Dashboard expects z-scored data
    'use_activation_normalization': False
}
```

---

## 4. Critical Issues

### 4.1 🔴 CRITICAL: Normalization Inconsistency

**Problem:** Dashboard expects z-score normalized probe scores, but this normalization happens at runtime in analysis scripts, not during preprocessing.

**Evidence:**
```python
# probe_configs.py:104
BASELINE_CONFIG = {
    'normalize_probe_scores': True,  # Dashboard expects this
}

# But in preprocessing scripts, normalization happens AFTER probe application
# If preprocessing script doesn't normalize, dashboard gets raw scores!
```

**Impact:**
- If preprocessed data wasn't created with `normalize_probe_scores=True`, dashboard displays raw probe logits
- Raw logits and z-scores have completely different scales (logits ~[-10, +10], z-scores ~[-3, +3])
- User sees incorrect emotion intensities and comparisons

**Recommendation:**
1. Store normalization metadata in preprocessed file
2. Add validation check on dashboard load
3. Consider storing both raw and normalized scores

### 4.2 🔴 CRITICAL: Probe Type Naming Confusion

**Problem:** `probe_type="linear"` vs `probe_type="standard"` refer to the same probes with different loading logic.

**Evidence:**
```python
# token_level_helpers.py:328
if self.probe_type == "linear":
    scores = self.inference.predict(...)  # Uses ProbeInference.predict()

# model_diff_helpers.py:340
elif self.probe_type == "standard":
    # Manually load pickle and apply
    with open(probe_path, 'rb') as f:
        probe_dict = pickle.load(f)
    probe_model = probe_dict['model']
    logits = probe_model(X_tensor)
```

**Impact:**
- Same probes might produce slightly different results due to different inference paths
- Confusion when setting up experiments ("which probe type do I use?")

**Recommendation:**
- Unify naming: use "standard" everywhere, or "linear" everywhere
- Remove redundant loading logic - use ProbeInference for all types

### 4.3 ⚠️ HIGH: Activation Caching Bypasses Stage 1 Normalization

**Problem:** When using cached activations, Stage 1 normalization is skipped entirely.

**Evidence:**
```python
# model_diff_helpers.py:160-172
if cached_activations is not None:
    if verbose:
        print("\n[2/4] Skipping normalization (using cached activations)")
elif self.use_wildchat_normalization:
    # Only normalizes if NOT using cached activations
    activations = self._normalize_activations(...)
```

**Impact:**
- If exp1 uses `use_wildchat_normalization=False`
- And exp2 reuses activations with `use_wildchat_normalization=True`
- exp2 will NOT get normalized activations despite the setting
- Results will be inconsistent with expectations

**Current Behavior:**
```python
# model_diff_analysis_v2.py:218
results_text_raw = exp2.run_experiment(
    ...,
    cached_activations=results_ortho_conv['activations']  # Uses exp1's unnormalized acts
)
```

**Recommendation:**
- Store normalization state with cached activations
- Validate that cached activations match requested normalization
- Or: always cache raw activations, apply normalization on load

### 4.4 ⚠️ HIGH: Token-Level Returns Raw Activations Despite Normalization

**Problem:** Token-level experiment normalizes activations internally but returns raw activations for caching.

**Evidence:**
```python
# token_level_helpers.py:172-174
import copy
raw_activations_by_token = copy.deepcopy(activations_by_token)

# ... normalization happens to activations_by_token ...

# token_level_helpers.py:290-291
return {
    'activations_by_token': raw_activations_by_token,  # Returns raw, not normalized!
```

**Impact:**
- Cached activations are always raw
- Next experiment using cached activations will apply normalization again
- This is actually **good behavior** (unlike model_diff), but inconsistent

**Recommendation:**
- Document this behavior clearly
- Make model_diff match this pattern (always cache raw)

---

## 5. Medium Priority Issues

### 5.1 ⚠️ Hardcoded Paths Throughout

**Locations:**
- `model_diff_analysis_v2.py`: Lines 47-48, 98, 110, 169-170, 201-202
- `token_level_experiment_v2.py`: Lines 54-55, 115, 144-145, 178-179, 210-211
- `eval_dashboard/app.py`: Line 464
- `probe_configs.py`: Lines 10-12

**Recommendation:**
- Create a `paths_config.py` with all path constants
- Use environment variables for base paths
- Add validation to check paths exist

### 5.2 ⚠️ Multiple Normalization Flags with Unclear Precedence

**Current flags:**
```python
use_wildchat_normalization    # Stage 1 (activations)
normalize_probe_scores         # Stage 2 (probe outputs, z-score)
center_probe_scores           # Stage 2 (probe outputs, centering only)
```

**Precedence rules:**
```python
# Stage 2 precedence (token_level_helpers.py:209-214)
if self.normalize_probe_scores or self.center_probe_scores:
    if self.normalize_probe_scores:
        # Use z-score (takes precedence)
    else:
        # Use centering
```

**Issues:**
- Not immediately clear what happens if both are True
- `center_probe_scores` marked as "superseded" in comment but still in code
- No validation/warning if conflicting options set

**Recommendation:**
- Replace two flags with single enum: `ProbeNormalization.NONE | CENTERED | ZSCORE`
- Add deprecation warning for `center_probe_scores`
- Document precedence in docstrings

### 5.3 ⚠️ Centroid Probe Format Detection

**Current behavior:**
```python
# model_diff_helpers.py:314-323
if self.probe_type == "centroid":
    # Load test probe to detect format
    test_data = self.inference.load_centroid_probe(
        layer=layers[0],
        k_value=self.k_value,
        orthogonality_weight=self.orthogonality_weight,
        constraint_type=self.centroid_constraint_type,
        probe_format=self.centroid_probe_format  # Can be "auto"
    )
    track_roles = test_data['probe_format'] == 'conversation'
```

**Issues:**
- Loads probe twice (once for detection, once in loop)
- Format detection happens in `load_centroid_probe` but not documented
- Unclear what happens if format is ambiguous

**Recommendation:**
- Cache format detection result
- Document format detection logic
- Consider storing format in probe filename

### 5.4 ⚠️ Layer-Averaged Baseline with `-1` Index

**Current behavior:**
```python
# model_diff_helpers.py:528-529
baseline_mean = baseline_stats['mean'][-1]  # Layer-averaged
baseline_std = baseline_stats['std'][-1]
```

**Issues:**
- Magic index `-1` not documented in return type
- Unclear if baseline stats dict has per-layer AND aggregated stats
- No validation that `-1` key exists

**Recommendation:**
- Use explicit key like `'layer_averaged'` instead of `-1`
- Document baseline stats structure
- Add type hints

---

## 6. Low Priority Issues

### 6.1 Documentation Gaps

**Missing documentation:**
- What does "raw" vs "global_cpca_top20" mean for orthogonal_representation?
- How are cPCA components computed and cached?
- What's the difference between "text_based" and "conversation_based" probe training?
- Why different orthogonality weights (1000 vs 100000)?

**Recommendation:**
- Add detailed docstring to `DoubleDiffExperiment` and `TokenLevelExperiment`
- Create a "Probe Training Guide" document
- Add inline comments explaining weight choices

### 6.2 Inconsistent Emotion Ordering

**Current:**
```python
EMOTIONS = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
```

**Issues:**
- No validation that probe outputs match this ordering
- If probe was trained with different emotion order, results will be misaligned
- No explicit mapping in probe files

**Recommendation:**
- Store emotion ordering in probe pickle files
- Validate ordering matches at load time
- Consider using dict outputs from probes instead of arrays

### 6.3 No Probe Version Tracking

**Issues:**
- No way to know which probe version was used in preprocessed data
- If probes are retrained, old preprocessed data becomes invalid
- No checksum or timestamp validation

**Recommendation:**
- Add probe metadata to preprocessed files
- Include probe file hash in results
- Add version field to probe pickle files

---

## 7. Recommendations Summary

### Immediate Actions (Critical)

1. **Fix Normalization Pipeline:**
   ```python
   # Store normalization metadata in preprocessed data
   {
       'conversations': [...],
       'normalization': {
           'probe_scores': {
               'method': 'zscore',  # or 'centered' or 'none'
               'baseline_mean': [...],
               'baseline_std': [...]
           },
           'activations': {
               'method': 'wildchat',  # or 'none'
               'aggregation': 'assistant_turn'
           }
       }
   }
   ```

2. **Unify Probe Type Naming:**
   ```python
   # Choose ONE naming convention
   PROBE_TYPES = {
       'orthogonal': 'Orthogonal conversation probes',
       'linear': 'Linear probes with cPCA projection',
       'centroid': 'K-averaged centroid probes'
   }
   # Remove "standard" entirely, use "linear" for multiseed probes
   ```

3. **Fix Activation Caching:**
   ```python
   # Always cache raw activations
   def run_experiment(..., cached_activations=None):
       if cached_activations is not None:
           # Validate cached activations are raw
           assert cached_activations['normalization'] == 'none'
           activations = cached_activations['data']
       else:
           activations = extract_activations(...)

       # Apply normalization after cache check
       if use_wildchat_normalization:
           activations = normalize(activations)
   ```

### Short-term Improvements (High Priority)

4. **Centralize Configuration:**
   ```python
   # paths_config.py
   from pathlib import Path
   import os

   BASE_DIR = Path(os.environ.get('RESEARCH_TOOLS_DIR', '/workspace-vast/annas/git/research-tools'))
   PROBE_DIR = BASE_DIR / 'outputs/probes/emotion_probes'
   BASELINE_DIR = BASE_DIR / 'data/baselines/alpaca_gemma27b_v2/google_gemma_3_27b_it'
   ```

5. **Replace Multiple Normalization Flags:**
   ```python
   from enum import Enum

   class ProbeNormalization(Enum):
       NONE = 'none'
       CENTERED = 'centered'  # x - mean
       ZSCORE = 'zscore'      # (x - mean) / std

   def __init__(self, ..., probe_normalization: ProbeNormalization = ProbeNormalization.ZSCORE):
       ...
   ```

6. **Add Validation Checks:**
   ```python
   # Dashboard load
   def load_preprocessed_data(data_path):
       data = pickle.load(...)

       # Validate normalization
       if 'normalization' not in data:
           st.warning("⚠️ Loaded data has no normalization metadata. Assuming raw scores.")
       elif data['normalization']['probe_scores']['method'] != 'zscore':
           st.error("❌ Dashboard expects z-score normalized data!")

       return data
   ```

### Long-term Improvements (Medium/Low Priority)

7. **Add Probe Versioning:**
   ```python
   # When saving probes
   probe_data = {
       'model': probe_model,
       'metadata': {
           'version': '1.0.0',
           'emotions': EMOTIONS,
           'trained_on': 'wildchat_v2',
           'timestamp': datetime.now().isoformat(),
           'hash': compute_hash(probe_model)
       }
   }
   ```

8. **Improve Documentation:**
   - Create `docs/PROBE_TYPES.md` explaining each probe type
   - Create `docs/NORMALIZATION_GUIDE.md` explaining both stages
   - Add architecture diagrams to README

9. **Add Integration Tests:**
   ```python
   def test_probe_output_format():
       """Verify orthogonal probes return nested dict format"""
       exp = DoubleDiffExperiment(..., probe_type='orthogonal')
       results = exp.run_experiment(...)

       sample_score = results['probe_scores']['ft_dataset'][30][0]
       assert isinstance(sample_score, dict)
       assert 'user' in sample_score
       assert 'assistant' in sample_score
       assert sample_score['user'].shape == (6,)
   ```

---

## 8. Conclusion

The probe application system is well-architected with a unified inference pipeline and flexible normalization options. However, there are critical issues around:

1. **Normalization consistency** between preprocessing and dashboard
2. **Activation caching** behavior that bypasses stage 1 normalization
3. **Probe type naming** confusion between "linear" and "standard"

Addressing these issues will ensure consistent results across analysis scripts and correct visualization in the dashboard.

**Priority Order:**
1. Fix normalization pipeline (Critical - affects all results)
2. Fix activation caching (Critical - affects cached experiments)
3. Unify probe type naming (High - affects usability)
4. Centralize paths (High - affects maintainability)
5. Add validation checks (Medium - prevents user errors)
6. Improve documentation (Low - affects onboarding)
