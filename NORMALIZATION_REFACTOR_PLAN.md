# Probe Normalization Refactoring Plan

## Goal
Simplify the normalization pipeline by:
1. **Remove** `use_wildchat_normalization` (activation-level normalization)
2. **Remove** `center_probe_scores` (centering-only mode)
3. **Always apply** z-score normalization to probe scores

## Rationale

### Current Problems
- 3 different normalization flags create confusion
- Inconsistent defaults between scripts
- Dashboard expects z-scored data but this isn't enforced
- Activation caching bypasses stage 1 normalization

### Benefits of Simplification
- **Consistency**: All systems use same normalization (z-scores in σ units)
- **Interpretability**: Z-scores are standardized and comparable across experiments
- **Simplicity**: No configuration needed - normalization always happens
- **Correctness**: Dashboard always receives expected format

## New Pipeline

```
Raw Activations (from model)
      ↓
[Apply Probes] ← unchanged
      ↓
Raw Probe Scores (logits)
      ↓
[Z-Score Normalization] ← ALWAYS applied, not configurable
      ↓  scores = (raw_scores - baseline_mean) / baseline_std
      ↓  where baseline_mean, baseline_std computed from WildChat
Final Scores (in standard deviations σ)
```

## Files to Modify

### Core Implementation Files (3)

#### 1. `probes/scripts/model_diff_helpers.py`
**Changes:**
```python
# REMOVE these parameters from __init__:
- use_wildchat_normalization: bool = False
- center_probe_scores: bool = False
- normalize_probe_scores: bool = False  # Always True now

# REMOVE Step 2 (activation normalization) entirely:
- _normalize_activations() method
- Lines 160-172 conditional logic

# SIMPLIFY Step 4 (probe score normalization):
- Remove conditional check (always normalize)
- Remove centering branch
- Always call normalize_probe_scores_zscore()
```

**New signature:**
```python
def __init__(
    self,
    base_model,
    ft_model,
    tokenizer,
    probe_type: str = "orthogonal",
    probe_dir: Path = None,
    cpca_path: Path = None,
    probe_pattern: str = None,
    orthogonality_weight: float = 1000.0,
    orthogonal_representation: str = "raw",
    n_components: int = 10,
    seed: int = 0,
    baseline_dir: Path = None,  # Still needed for computing baseline stats
    emotions: List[str] = None,
    k_value: Optional[int] = None,
    centroid_constraint_type: Optional[str] = None,
    centroid_probe_format: str = "auto"
):
    """
    Initialize experiment configuration.

    Note: Probe scores are ALWAYS z-score normalized using WildChat baseline statistics.
    This ensures consistent, interpretable scores in standard deviation (σ) units.
    """
```

#### 2. `probes/scripts/token_level_helpers.py`
**Changes:**
```python
# REMOVE these parameters from __init__:
- use_wildchat_normalization: bool = False
- wildchat_aggregation: str = "assistant_turn"
- center_probe_scores: bool = False
- normalize_probe_scores: bool = False  # Always True now

# REMOVE Step 2 (activation normalization) entirely:
- Lines 177-196 conditional logic
- baseline_loader.normalize_token_level() call

# SIMPLIFY Step 4 (probe score normalization):
- Remove conditional check (always normalize)
- Remove centering branch
- Always call normalize_probe_scores_zscore()
```

#### 3. `probes/scripts/probe_pipeline.py` (if it exists)
**Check if these functions can be simplified:**
- `normalize_probe_scores_zscore()` - Keep this
- `normalize_probe_scores_center()` - Can remove this

### Analysis Scripts (3)

#### 4. `probes/scripts/model_diff_analysis_v2.py`
**Changes:**
```python
# REMOVE configuration section (lines 107-110):
- USE_BASELINE_NORMALIZATION = False
- NORMALIZE_PROBE_SCORES = True
- BASELINE_DIR = Path(...)  # Keep this, but rename

# UPDATE experiment initialization (remove flags):
exp1 = DoubleDiffExperiment(
    base_model=base_model,
    ft_model=ft_model,
    tokenizer=tokenizer,
    probe_type="orthogonal",
    probe_dir=Path(".../conversation_based/"),
    cpca_path=Path(".../gemma-3-27b-it_cpca.npz"),
    orthogonality_weight=1000.0,
    orthogonal_representation="raw",
    baseline_dir=BASELINE_DIR  # Only baseline-related param
)
```

#### 5. `probes/scripts/token_level_experiment_v2.py`
**Changes:**
```python
# REMOVE configuration section (lines 111-118):
- USE_BASELINE_NORMALIZATION = False
- NORMALIZE_PROBE_SCORES = True
- CENTER_PROBE_SCORES = False
- BASELINE_AGGREGATION = "all_tokens"  # Can hardcode this in implementation

# UPDATE experiment initialization (remove flags):
exp1 = TokenLevelExperiment(
    model=model,
    tokenizer=tokenizer,
    probe_type="orthogonal",
    probe_dir=Path(".../conversation_based/"),
    cpca_path=Path(".../gemma-3-27b-it_cpca.npz"),
    orthogonality_weight=1000.0,
    orthogonal_representation="raw",
    baseline_dir=BASELINE_DIR
)
```

#### 6. `elicitation/scripts/run_emotion_onset_probes.py`
**Similar changes** - remove normalization flags from experiment initialization

### Dashboard & Preprocessing (2)

#### 7. `eval_dashboard/probe_configs.py`
**Changes:**
```python
# UPDATE baseline config (lines 100-106):
BASELINE_CONFIG = {
    'baseline_dir': RESEARCH_TOOLS / "data/baselines/alpaca_gemma27b_v2/google_gemma_3_27b_it",
    # Remove these (implicit now):
    # 'normalize_probe_scores': True,
    # 'use_activation_normalization': False,
}

# Add comment:
# Note: All probe scores are z-score normalized using WildChat baseline statistics.
# Scores are in standard deviation (σ) units relative to baseline distribution.
```

#### 8. `eval_dashboard/data_preprocessing.py`
**Changes:**
- Update to use new API (remove normalization flags)
- Add assertion that scores are normalized (for validation)

### Documentation (4)

#### 9. `probes/scripts/PROBE_SCORE_CENTERING.md`
**Action:** Delete or mark as deprecated
- This entire document describes centering option we're removing

#### 10. `probes/scripts/QUICK_START_CENTERING.md`
**Action:** Delete or rename to `QUICK_START.md`
- Remove centering instructions

#### 11. `probes/scripts/TOKEN_LEVEL_README.md`
**Action:** Update normalization section
- Remove discussion of normalization options
- Document that z-score normalization is always applied

#### 12. `probes/scripts/MODEL_DIFF_README.md`
**Action:** Update normalization section
- Same as TOKEN_LEVEL_README.md

#### 13. `PROBE_APPLICATION_AUDIT.md`
**Action:** Update with new architecture
- Mark as resolved
- Document simplified pipeline

### Example Scripts (1)

#### 14. `probes/scripts/multi_experiment_example.py`
**Action:** Update examples to remove flags

## Implementation Steps

### Phase 1: Core Implementation (Critical)
1. ✅ Update `model_diff_helpers.py`
   - Remove `use_wildchat_normalization`, `center_probe_scores` params
   - Remove `_normalize_activations()` method
   - Simplify `_normalize_probe_scores()` to always use z-score
   - Update docstrings

2. ✅ Update `token_level_helpers.py`
   - Same changes as model_diff_helpers.py
   - Remove wildchat_aggregation param (hardcode to sensible default)

3. ✅ Update `probe_pipeline.py` (if needed)
   - Remove `normalize_probe_scores_center()` function

### Phase 2: Update Scripts (High Priority)
4. ✅ Update `model_diff_analysis_v2.py`
   - Remove configuration flags
   - Update experiment initialization

5. ✅ Update `token_level_experiment_v2.py`
   - Remove configuration flags
   - Update experiment initialization

6. ✅ Update `run_emotion_onset_probes.py`
   - Remove normalization flags

### Phase 3: Dashboard Integration (High Priority)
7. ✅ Update `probe_configs.py`
   - Simplify baseline config
   - Add documentation comment

8. ✅ Update `data_preprocessing.py`
   - Use new API
   - Add validation checks

### Phase 4: Documentation (Medium Priority)
9. ✅ Delete/deprecate centering documentation
10. ✅ Update README files
11. ✅ Update audit report

### Phase 5: Testing (High Priority)
12. ✅ Run model_diff experiment
13. ✅ Run token_level experiment
14. ✅ Regenerate preprocessed dashboard data
15. ✅ Verify dashboard displays correctly

## Baseline Statistics Configuration

Even though we're removing normalization flags, we still need baseline statistics.

**Hardcoded defaults:**
```python
# For probe score normalization (ALWAYS applied)
DEFAULT_BASELINE_DIR = Path("/workspace-vast/annas/git/research-tools/data/baselines/alpaca_gemma27b_v2/google_gemma_3_27b_it")

# Aggregation type for computing baseline statistics
# (user doesn't need to configure this anymore)
BASELINE_AGGREGATION = "all_tokens"  # Most general aggregation
```

**What gets computed from baselines:**
```python
# For each layer and each emotion:
baseline_mean = np.mean(baseline_probe_scores)  # Mean probe output on WildChat
baseline_std = np.std(baseline_probe_scores)    # Std dev of probe outputs

# Then normalize:
normalized_score = (raw_score - baseline_mean) / baseline_std
```

## Migration Guide for Users

### Before (Old API)
```python
exp = DoubleDiffExperiment(
    base_model=base_model,
    ft_model=ft_model,
    tokenizer=tokenizer,
    probe_type="orthogonal",
    probe_dir=probe_dir,
    use_wildchat_normalization=False,    # ← REMOVE
    normalize_probe_scores=True,         # ← REMOVE (always True)
    center_probe_scores=False,           # ← REMOVE
    baseline_dir=baseline_dir
)
```

### After (New API)
```python
exp = DoubleDiffExperiment(
    base_model=base_model,
    ft_model=ft_model,
    tokenizer=tokenizer,
    probe_type="orthogonal",
    probe_dir=probe_dir,
    baseline_dir=baseline_dir  # Still needed for computing stats
)
# Probe scores are automatically z-score normalized using WildChat baselines
```

## Validation Checklist

After refactoring, verify:
- [ ] Model diffing experiments run without errors
- [ ] Token-level experiments run without errors
- [ ] Dashboard preprocessing script works
- [ ] Dashboard displays correctly with z-scored data
- [ ] All probe types work (orthogonal, linear, centroid)
- [ ] Cached activation reuse still works
- [ ] Output scores are in reasonable range (typically -3 to +3 σ)
- [ ] Double-diff statistics are sensible
- [ ] Documentation is updated

## Backward Compatibility

**Breaking changes:**
- Old scripts with normalization flags will fail with `TypeError: unexpected keyword argument`
- Preprocessed dashboard data from old pipeline may have different scales

**Migration strategy:**
1. Add deprecation warnings if flags are passed (optional grace period)
2. Regenerate all preprocessed dashboard data
3. Update all example scripts in repo
4. Add version tag to preprocessed files for validation

## Questions to Resolve

1. **Baseline aggregation type**: Should we hardcode `"all_tokens"` or keep it configurable?
   - Recommendation: Hardcode to `"all_tokens"` (most general)

2. **Baseline directory**: Should this be a parameter or environment variable?
   - Recommendation: Keep as parameter, add default value

3. **Layer-specific vs layer-averaged baselines**: Currently uses layer-averaged (`baseline_stats[-1]`)
   - Recommendation: Keep layer-averaged (simpler, more robust)

4. **What if baseline_dir doesn't exist?**: Should it fail or warn?
   - Recommendation: Fail with clear error message

## Success Metrics

After refactoring:
- ✅ Fewer parameters (removed 3 flags)
- ✅ Zero normalization-related configuration needed
- ✅ All probe scores in consistent units (σ)
- ✅ Dashboard always receives expected format
- ✅ No more "which normalization should I use?" questions
- ✅ Simpler onboarding for new users
