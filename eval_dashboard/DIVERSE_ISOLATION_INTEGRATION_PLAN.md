# Plan: Adding Diverse Isolation Probes to Eval Dashboard

## Overview

Add diverse isolation probes (user and assistant, with opposite-source regularization) to the emotion onset evaluation dashboard, aggregated over layers 30-40 with separate user/assistant plots.

## Current Dashboard Architecture

### Probe Configuration System (probe_configs.py)
- Dictionary-based configuration: `PROBE_CONFIGS[probe_key] = {...}`
- Key fields:
  - `type`: Probe type (linear, orthogonal, centroid, orthogonal_regularized)
  - `probe_dir`: Directory containing probe files
  - `split_user_asst`: Boolean - if True, shows separate user/assistant lines
  - `color_user`, `color_asst`: Colors for split probes
  - `probe_pattern`: Custom filename pattern (optional)

### Preprocessing Pipeline (data_preprocessing.py)
1. Load emotion onset data
2. Extract activations for each conversation
3. Initialize probe experiments (TokenLevelExperiment)
4. Apply probes to get token-level scores
5. Compute baseline statistics for z-score normalization
6. Aggregate scores to sentence level
7. Save preprocessed data with all probe scores

### Probe Loading (probe_pipeline.py: ProbeInference)
- Supports 4 probe types: linear, orthogonal, centroid, orthogonal_regularized
- Each type has dedicated load/predict methods
- Probes are cached for efficiency

### Existing Similar Pattern: Orthogonal Regularized Probes
- Uses `add_orthogonal_regularized_probes.py` script
- Loads model from pickle files
- Uses existing TokenLevelExperiment infrastructure
- Single-perspective (no user/assistant split)

## Diverse Isolation Probe Details

### File Structure
```
outputs/probes/diverse_isolation/
  ├── user_layer{layer}_lambda{lambda}/
  │   ├── model.pt              # PyTorch state dict
  │   └── results.json          # Training results
  └── assistant_layer{layer}_lambda{lambda}/
      ├── model.pt
      └── results.json
```

### Model Format
- PyTorch checkpoint (`.pt` file)
- Contains: `SourceSpecificEmotionProbe` (simple Linear layer + optional orthogonal PCs)
- State dict keys: `linear.weight`, `linear.bias`, `orthogonal_pcs` (optional)

### Target Configuration
- Lambda value: 10.0 (best performing based on recent results)
- Layers: 30-40 (11 layers to aggregate)
- Both isolation types: user and assistant
- Display: Separate user/assistant lines (like orthogonal probes)

## Implementation Plan

### 1. Add New Probe Type to ProbeInference

**File**: `probes/scripts/probe_pipeline.py`

**Method to add**: `load_diverse_isolation_probe()`
```python
def load_diverse_isolation_probe(
    self,
    layer: int,
    isolation_type: str,  # 'user' or 'assistant'
    lambda_reg: float = 10.0
) -> Dict:
    """
    Load diverse isolation probe (user or assistant specific).

    Returns:
        Dictionary with 'model' (SourceSpecificEmotionProbe) and 'label_names'
    """
```

**Method to add**: `predict_diverse_isolation()`
```python
def predict_diverse_isolation(
    self,
    activations: np.ndarray,
    probe_model,
    emotions: list
) -> np.ndarray:
    """
    Apply diverse isolation probe to activations.

    Returns:
        Scores array [batch_size, n_emotions]
    """
```

**Integration point in TokenLevelExperiment._apply_probes()**:
- Add case for `probe_type == 'diverse_isolation'`
- Load both user and assistant probes per layer
- Return dict format: `{'user': user_scores, 'assistant': asst_scores}`

### 2. Add Probe Configs

**File**: `eval_dashboard/probe_configs.py`

**Two options**:

**Option A**: Combined user+assistant in one config (RECOMMENDED)
```python
'diverse_isolation_lambda10': {
    'name': 'Diverse Isolation (User/Asst) - Lambda 10',
    'display_name': 'Diverse Isolation λ=10',
    'type': 'diverse_isolation',
    'probe_dir': RESEARCH_TOOLS / 'outputs/probes/diverse_isolation',
    'lambda_reg': 10.0,
    'split_user_asst': True,
    'color_user': '#2980b9',  # Deep blue for user
    'color_asst': '#c0392b',  # Deep red for assistant
    'description': 'Source-specific probes with opposite-source regularization (λ=10)'
}
```

**Option B**: Separate configs for user and assistant
```python
'diverse_isolation_user_lambda10': {
    'name': 'Diverse Isolation (User) - Lambda 10',
    'display_name': 'Diverse Iso User λ=10',
    'type': 'diverse_isolation',
    'probe_dir': RESEARCH_TOOLS / 'outputs/probes/diverse_isolation',
    'isolation_type': 'user',
    'lambda_reg': 10.0,
    'split_user_asst': False,
    'color': '#2980b9',
    'description': 'User emotion probe with assistant regularization (λ=10)'
},
'diverse_isolation_assistant_lambda10': {
    'name': 'Diverse Isolation (Assistant) - Lambda 10',
    'display_name': 'Diverse Iso Asst λ=10',
    'type': 'diverse_isolation',
    'probe_dir': RESEARCH_TOOLS / 'outputs/probes/diverse_isolation',
    'isolation_type': 'assistant',
    'lambda_reg': 10.0,
    'split_user_asst': False,
    'color': '#c0392b',
    'description': 'Assistant emotion probe with user regularization (λ=10)'
}
```

**Recommendation**: Use Option A (combined config) to match the orthogonal probe pattern and enable direct comparison.

### 3. Create Preprocessing Script

**File**: `eval_dashboard/add_diverse_isolation_probes.py`

**Based on**: `add_orthogonal_regularized_probes.py`

**Key modifications**:
1. Load both user and assistant probes per layer
2. Apply probes separately to activations
3. Return dict format: `{'user': user_scores, 'assistant': asst_scores}`
4. Aggregate across layers 30-40 (mean)
5. Apply z-score normalization with baseline stats
6. Aggregate to sentence level
7. Add to existing preprocessed data

**Steps**:
```python
# 1. Load existing preprocessed data
# 2. Load model and tokenizer
# 3. Compute baseline stats for both user and assistant probes
# 4. For each conversation:
#    - Re-extract activations
#    - Apply user probe (layers 30-40, aggregate)
#    - Apply assistant probe (layers 30-40, aggregate)
#    - Normalize scores
#    - Return dict: {'user': user_scores, 'assistant': asst_scores}
#    - Aggregate to sentences
# 5. Save updated data
```

### 4. Update TokenLevelExperiment

**File**: `probes/scripts/token_level_helpers.py`

**Modifications**:

1. Add `isolation_type` and `lambda_reg` parameters to `__init__`
2. Update `_apply_probes()` to handle `probe_type == 'diverse_isolation'`:
   ```python
   elif self.probe_type == 'diverse_isolation':
       # Load both user and assistant probes
       for layer in layers:
           user_probe = self.inference.load_diverse_isolation_probe(
               layer, 'user', self.lambda_reg
           )
           asst_probe = self.inference.load_diverse_isolation_probe(
               layer, 'assistant', self.lambda_reg
           )

           # Apply both probes
           user_scores = self.inference.predict_diverse_isolation(...)
           asst_scores = self.inference.predict_diverse_isolation(...)

           # Store as dict
           scores_by_token[token_pos][layer] = {
               'user': user_scores,
               'assistant': asst_scores
           }
   ```

### 5. Update Baseline Loader

**File**: `probes/scripts/wildchat_baseline_loader.py`

**Modification**: Add support for `diverse_isolation` probe type in `compute_probe_score_baselines()`

**Approach**:
- Similar to orthogonal probes - compute separate baselines for user and assistant
- Return dict: `{'mean': {'user': [...], 'assistant': [...]}, 'std': {'user': [...], 'assistant': [...]}}`

## Testing Strategy

1. **Probe Loading Test**:
   ```python
   # Test loading single probe
   from probes.scripts.probe_pipeline import ProbeInference

   inference = ProbeInference(
       probe_dir=Path('outputs/probes/diverse_isolation'),
       cpca_path=None,
       device='cuda'
   )

   probe = inference.load_diverse_isolation_probe(
       layer=30, isolation_type='user', lambda_reg=10.0
   )
   print(probe['model'])
   ```

2. **Baseline Computation Test**:
   ```python
   # Verify baseline stats compute correctly
   from eval_dashboard.data_preprocessing import *

   # Test with small subset of baseline data
   ```

3. **Preprocessing Test**:
   ```bash
   # Run on small subset first
   python eval_dashboard/add_diverse_isolation_probes.py --test-mode
   ```

4. **Dashboard Verification**:
   ```bash
   # Launch dashboard and verify new probe appears
   streamlit run eval_dashboard/app.py
   ```

## File Modifications Summary

### New Files:
1. `eval_dashboard/add_diverse_isolation_probes.py` - Preprocessing script
2. `eval_dashboard/DIVERSE_ISOLATION_INTEGRATION_PLAN.md` - This document

### Modified Files:
1. `probes/scripts/probe_pipeline.py`
   - Add `load_diverse_isolation_probe()`
   - Add `predict_diverse_isolation()`

2. `probes/scripts/token_level_helpers.py`
   - Add `isolation_type` and `lambda_reg` parameters
   - Add diverse_isolation case in `_apply_probes()`

3. `eval_dashboard/probe_configs.py`
   - Add diverse_isolation config(s)

4. `probes/scripts/wildchat_baseline_loader.py`
   - Add diverse_isolation support in `compute_probe_score_baselines()`

5. `eval_dashboard/data_preprocessing.py`
   - Update to handle new probe type (may not need changes if using existing infrastructure)

## Alternative: Simpler Approach

If modifying core probe infrastructure is too complex, we can use a **standalone approach**:

1. Create custom loading functions in `add_diverse_isolation_probes.py`
2. Manually load PyTorch models and apply them to activations
3. Format output to match dashboard expectations
4. Skip TokenLevelExperiment integration

This matches the pattern used in `eval_diverse_isolation_cross_source.py`:
```python
def load_trained_probe(model_path: str, device: str = 'cuda') -> nn.Module:
    checkpoint = torch.load(model_path, map_location='cpu')
    hidden_dim = checkpoint['linear.weight'].shape[1]
    num_classes = checkpoint['linear.weight'].shape[0]
    model = SourceSpecificEmotionProbe(hidden_dim, num_classes, ...)
    model.load_state_dict(checkpoint)
    return model.to(device)
```

**Pros**:
- Faster implementation
- No risk of breaking existing probe infrastructure
- Self-contained code

**Cons**:
- Code duplication
- Harder to maintain
- Won't integrate with other tools that use TokenLevelExperiment

## Recommendation

**Use the Simpler Approach for initial integration**:
1. Create standalone `add_diverse_isolation_probes.py` with custom loading
2. Add probe config to `probe_configs.py` with `type='diverse_isolation_custom'`
3. Update dashboard to handle new type if needed
4. Later refactor into core infrastructure if widely used

## Color Palette Suggestion

To maintain visual consistency and distinguish from existing probes:

- **Diverse Isolation User**: `#2980b9` (Deep blue - contrasts with orthogonal user)
- **Diverse Isolation Assistant**: `#c0392b` (Deep red - contrasts with orthogonal assistant)

## Next Steps

1. Implement standalone preprocessing script
2. Add probe config
3. Test on small subset
4. Run full preprocessing
5. Verify in dashboard
6. Document usage
