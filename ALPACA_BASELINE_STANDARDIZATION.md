# Alpaca Baseline Standardization - Complete ✅

**Date:** 2026-01-01
**Status:** Successfully Completed
**Commit:** `0eb197a`

---

## Summary

Successfully standardized ALL baseline references to use **Alpaca V2** instead of WildChat. The entire codebase now defaults to the neutral instruction-following Alpaca baseline for probe score normalization.

---

## Changes Made

### 1. Core Baseline Loader (wildchat_baseline_loader.py)

**Default baseline directory changed:**
```python
# Before:
baseline_dir = Path(".../data/baselines/wildchat/google_gemma_3_27b_it")

# After:
baseline_dir = Path(".../data/baselines/alpaca_gemma27b_v2/google_gemma_3_27b_it")
```

**Default aggregation type changed:**
```python
# Before:
aggregation_type = "assistant_turn"

# After:
aggregation_type = "all_tokens"  # Matches actual usage everywhere
```

**Documentation updated:**
- Module docstring now clarifies it works with any baseline (Alpaca by default)
- Class docstring explains historical name but current Alpaca default
- Error messages made more generic (removed WildChat-specific text)

### 2. Documentation Updates

**MODEL_DIFF_README.md:**
- Added prominent deprecation notice at top
- Updated baseline path from WildChat to Alpaca V2
- Added explanation of why Alpaca is better

**TOKEN_LEVEL_README.md:**
- Added prominent deprecation notice at top
- Updated baseline reference from WildChat to Alpaca
- Pointed to current documentation (BASELINE_STATISTICS_GUIDE.md)

---

## Why Alpaca Instead of WildChat?

| Aspect | WildChat (Old) | Alpaca V2 (New) |
|--------|----------------|-----------------|
| **Content** | Real user conversations | Instruction-following dataset |
| **Emotional tone** | Varied, sometimes emotional/toxic | Neutral, task-focused |
| **Baseline quality** | Noisy, high variance | Stable, low variance |
| **Use case** | General chat baseline | Neutral reference for emotion detection |

**Result:** Probe scores now measure **deviation from neutral instruction-following** rather than deviation from varied real conversations. This makes emotional content more detectable.

---

## Current Standard

**All systems now use:**
```
Location: /workspace-vast/annas/git/research-tools/data/baselines/alpaca_gemma27b_v2/google_gemma_3_27b_it/
Aggregation: all_tokens
Normalization: Always z-score (σ units)
```

**No configuration needed** - Alpaca baseline is automatic default.

---

## Verification

### Files Using Alpaca Baseline (Default or Explicit)

1. **Model Diffing:**
   - `probes/scripts/model_diff_analysis_v2.py:108` ✅
   - Explicit path to Alpaca V2

2. **Token-Level Analysis:**
   - `probes/scripts/token_level_experiment_v2.py:112` ✅
   - Explicit path to Alpaca V2

3. **Emotion Onset Probes:**
   - `elicitation/scripts/run_emotion_onset_probes.py:72` ✅
   - Explicit path to Alpaca V2

4. **Dashboard:**
   - `eval_dashboard/probe_configs.py:104` ✅
   - Explicit path to Alpaca V2
   - `eval_dashboard/data_preprocessing.py` ✅
   - Uses config from probe_configs.py

5. **Core Loader:**
   - `probes/scripts/wildchat_baseline_loader.py:28` ✅
   - **NOW defaults to Alpaca V2**

---

## Legacy WildChat Support

WildChat baseline generation scripts are **retained for backward compatibility** but not used by default:

- `probes/scripts/compute_wildchat_baseline_activations.py` - Kept for legacy
- `probes/scripts/check_wildchat_progress.sh` - Kept for legacy
- `probes/scripts/compute_probe_baselines_from_wildchat.py` - Kept for legacy

Users can still explicitly use WildChat by passing:
```python
baseline_dir=Path("data/baselines/wildchat/google_gemma_3_27b_it")
```

But **no code defaults to this anymore**.

---

## Related Work

This change completes the baseline standardization work:

1. **PROBE_APPLICATION_AUDIT.md** - Initial audit identifying baseline inconsistencies
2. **NORMALIZATION_REFACTOR_PLAN.md** - Plan for simplifying normalization
3. **NORMALIZATION_REFACTOR_COMPLETE.md** - Completed refactoring (always z-score)
4. **BASELINE_STATISTICS_GUIDE.md** - Comprehensive guide to both baseline systems
5. **ALPACA_BASELINE_STANDARDIZATION.md** (this doc) - Standardized to Alpaca everywhere

---

## Migration Guide

### For Users

**No action needed!** All experiments now automatically use Alpaca baselines.

### For Code Updates

If you have custom scripts that import WildChatBaselineLoader:

```python
# Old code (still works, but now uses Alpaca by default):
from probes.scripts.wildchat_baseline_loader import WildChatBaselineLoader
loader = WildChatBaselineLoader()  # Now defaults to Alpaca!

# Explicit Alpaca (recommended for clarity):
loader = WildChatBaselineLoader(
    baseline_dir=Path("data/baselines/alpaca_gemma27b_v2/google_gemma_3_27b_it")
)

# Explicit WildChat (if you really want legacy):
loader = WildChatBaselineLoader(
    baseline_dir=Path("data/baselines/wildchat/google_gemma_3_27b_it")
)
```

---

## Future Considerations

### Potential Cleanup (Not Required)

1. **Rename WildChatBaselineLoader → BaselineLoader**
   - Current name is confusing (not WildChat-specific)
   - Works with any baseline dataset
   - Would break imports, so low priority

2. **Archive WildChat Baselines**
   - If not actively used, could move to archive/
   - Save disk space (~several GB)
   - Keep generation scripts for reference

3. **Update Remaining READMEs**
   - TOKEN_LEVEL_README.md has deprecation notice but old examples
   - MODEL_DIFF_README.md has deprecation notice but old examples
   - Could fully rewrite or leave as historical reference

---

## Success Metrics

All targets achieved:

- ✅ **Default baseline**: Now Alpaca V2 everywhere
- ✅ **Explicit usage**: All active experiments reference Alpaca V2
- ✅ **No WildChat defaults**: Removed from all default parameters
- ✅ **Documentation**: Updated READMEs with deprecation notices and new paths
- ✅ **Backward compatibility**: WildChat still available if explicitly requested
- ✅ **Consistency**: Single source of truth for baseline statistics

---

## Commits

This work was completed in the following commits:

1. `3aede74` - Probe application audit and normalization refactor plan
2. `e2b8d90` - Refactor: Simplify probe normalization pipeline
3. `20d5e2f` - Fix: Resolve code ordering issues
4. `ad4f4ba` - Add comprehensive baseline statistics documentation
5. `5e7b1ce` - Fix remaining references to removed normalization parameters
6. **`0eb197a` - Switch all default baselines from WildChat to Alpaca V2** ⭐

---

## Conclusion

The research-tools codebase now has **consistent, neutral baseline normalization** using Alpaca V2:

- **One standard baseline** (not multiple competing options)
- **No configuration needed** (automatic z-score with Alpaca)
- **Clear documentation** (explains what, why, and where)
- **Easy to use** (just import and run)

All probe scores are now in comparable σ units relative to a neutral instruction-following baseline. 🎉
