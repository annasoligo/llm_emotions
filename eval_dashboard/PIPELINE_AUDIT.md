# Pipeline Audit: Post-Hoc Fixes and Integration Opportunities

## Executive Summary

The current preprocessing pipeline has **accumulated many post-hoc "add_*" scripts** that modify existing pickle files instead of being integrated into the initial preprocessing. This creates:

- ❌ **Inefficiency**: Multiple full model loads and forward passes
- ❌ **Error-prone**: Easy to forget steps or run in wrong order
- ❌ **Hard to maintain**: 13+ separate scripts to track
- ❌ **Slow iteration**: Must rerun multiple scripts when regenerating data

**Recommendation**: Consolidate post-hoc fixes into `data_preprocessing.py` and make all probe types available at preprocessing time.

---

## Post-Hoc Scripts Inventory

### Category 1: ✅ **FIXED** - Integrated Baseline Correction

| Script | Purpose | Status |
|--------|---------|--------|
| `apply_baseline_correction.py` | Remove correlation with mean logits | ✅ **OBSOLETE** - Integrated into logit/axis scripts |
| `add_logit_lens_to_existing.py` | Add logit lens scores | ✅ **UPDATED** - Now includes 2-pass correction |
| `add_axis_lens_to_existing.py` | Add axis lens scores | ✅ **UPDATED** - Now includes 2-pass correction |

**Impact**: Eliminated 1 manual step, integrated correction into preprocessing.

---

### Category 2: 🔴 **NEEDS INTEGRATION** - Probe Type Post-Addition

These scripts add probe types that should be available at initial preprocessing:

| Script | Purpose | Why Post-Hoc? | Integration Priority |
|--------|---------|----------------|---------------------|
| `add_orthogonal_regularized_probes.py` | Add orthogonal regularized λ=100 | Probe type added after initial preprocessing | **HIGH** |
| `add_text_probes.py` | Add text-based probes | Not included in initial probe set | **HIGH** |
| `add_diverse_isolation_probes.py` | Add diverse isolation probes | Experimental probe type | **MEDIUM** |
| `add_logit_lens_layers.py` | Add additional layer ranges (L20-30, L30-40) | Initial only had L40-50 | **HIGH** |
| `add_logit_lens_all_tabs.py` | Add logit lens to all subsets | Batch script | **LOW** (wrapper) |

**Root cause**: `data_preprocessing.py` doesn't include all probe types in default config.

**Solution**: Add all probe types to `probe_configs.py` and accept them as arguments to `data_preprocessing.py`.

---

### Category 3: 🟡 **WORKFLOW HACKS** - Should Be Eliminated

These scripts exist to work around workflow inefficiencies:

| Script | Purpose | Why It Exists | Solution |
|--------|---------|---------------|----------|
| `copy_orthogonal_scores.py` | Copy scores from one file to others | Avoid recomputing (expensive) | **Preprocess all subsets at once** |
| `remove_orthogonal_scores.py` | Remove scores from files | Fix mistakes | **Better validation** |
| `recompute_logit_incremental.py` | Recompute with checkpoints | Fix corrupted data | **Prevent corruption** |
| `fix_missing_logit_data.sh` | Fix missing layer ranges | Incomplete preprocessing | **Complete preprocessing first time** |

**These should not exist** - they're band-aids for workflow problems.

---

### Category 4: ✅ **ANALYSIS/UTILITY** - Keep As-Is

These are legitimate one-off analysis scripts:

| Script | Purpose | Status |
|--------|---------|--------|
| `check_probe_correlations.py` | Analyze probe agreement | ✅ Keep |
| `analyze_probe_agreement.py` | Statistical analysis | ✅ Keep |
| `summarize_scenario_effects.py` | Summarize results | ✅ Keep |
| `test_*.py` | Unit tests | ✅ Keep |

---

## Current Workflow (Problematic)

```
1. Initial Preprocessing
   └─ data_preprocessing.py (orthogonal_raw, text_cpca, centroid_k10)
      ├─ Loads model
      ├─ Extracts activations
      └─ Saves: subset1.pkl, subset2.pkl, ...

2. Add Orthogonal Regularized (Manual Step)
   └─ add_orthogonal_regularized_probes.py
      ├─ Loads model AGAIN
      ├─ Extracts activations AGAIN
      └─ Updates all pickle files

3. Add Text Probes (Manual Step)
   └─ add_text_probes.py
      ├─ Loads model AGAIN
      ├─ Extracts activations AGAIN
      └─ Updates all pickle files

4. Add Logit Lens (Manual Step)
   └─ add_logit_lens_to_existing.py
      ├─ Loads model AGAIN
      ├─ Extracts activations AGAIN
      └─ Updates all pickle files

5. Add Axis Lens (Manual Step)
   └─ add_axis_lens_to_existing.py
      ├─ Loads model AGAIN
      ├─ Extracts activations AGAIN
      └─ Updates all pickle files

6. Add More Logit Lens Layers (Manual Step)
   └─ add_logit_lens_layers.py --layer_start 20 --layer_end 30
      ├─ Loads model AGAIN
      ├─ Extracts activations AGAIN
      └─ Updates all pickle files

7. Apply Baseline Correction (Manual Step) [NOW OBSOLETE]
   └─ apply_baseline_correction.py
      ├─ Loads pickle
      ├─ Computes alpha
      └─ Updates all pickle files
```

**Problem**: Model loaded 6+ times, activations extracted 6+ times for same conversations!

---

## Proposed Workflow (Efficient)

```
1. Single Preprocessing Run
   └─ data_preprocessing.py --all-probes --all-layers
      ├─ Loads model ONCE
      ├─ Extracts activations ONCE
      ├─ Applies ALL probe types:
      │  ├─ orthogonal_raw
      │  ├─ orthogonal_cpca_top10
      │  ├─ orthogonal_regularized_lambda100
      │  ├─ text_raw
      │  ├─ text_cpca
      │  ├─ centroid_k10
      │  ├─ diverse_isolation_lambda10
      │  ├─ logit_lens_mean (L40-50) + baseline correction
      │  ├─ logit_lens_mean_l30_40 + baseline correction
      │  ├─ logit_lens_mean_l20_30 + baseline correction
      │  ├─ axis_lens_mean + baseline correction
      │  └─ ... any other probes
      └─ Saves: Complete pickle files, one preprocessing run

2. Analysis (As Needed)
   └─ analyze_probe_agreement.py, check_correlations.py, etc.
```

**Benefits**:
- ✅ Model loaded ONCE
- ✅ Activations extracted ONCE
- ✅ No manual steps
- ✅ Consistent across all subsets
- ✅ Easy to add new conversations (just preprocess new ones)

---

## Implementation Plan

### Phase 1: Consolidate Probe Types ⚡ **HIGH PRIORITY**

**Goal**: All probe types available in `data_preprocessing.py`

**Steps**:
1. Add to `probe_configs.py`:
   - ✅ Already has: `orthogonal_raw`, `orthogonal_cpca_top10/20`, `text_raw`, `text_cpca`, `centroid_k10/k50`
   - ➕ Add: `orthogonal_regularized_lambda100` (currently only in separate script)
   - ➕ Add: `diverse_isolation_lambda10` (currently only in separate script)

2. Update `data_preprocessing.py`:
   - ✅ Already handles: All standard probe types
   - ➕ Add: Special handling for orthogonal_regularized (needs lambda parameter)
   - ➕ Add: Special handling for diverse_isolation

3. Create convenience script:
   ```bash
   #!/bin/bash
   # preprocess_all_probes.sh
   PROBES="orthogonal_raw orthogonal_cpca_top10 orthogonal_regularized_lambda100 \
           text_raw text_cpca centroid_k10 diverse_isolation_lambda10"

   python data_preprocessing.py \
       --input data.jsonl \
       --output data.pkl \
       --probes $PROBES
   ```

**Expected impact**: Eliminates 3 manual scripts (`add_orthogonal_regularized`, `add_text_probes`, `add_diverse_isolation`)

---

### Phase 2: Integrate Logit/Axis Lens into Main Preprocessing 🔥 **MEDIUM PRIORITY**

**Goal**: Logit lens and axis lens computed during initial preprocessing (not post-hoc)

**Challenge**: These use different codebases (emo_lens vs probes)

**Option A: Keep Separate (Current)**
- ✅ Pros: Clean separation, emo_lens code unchanged
- ❌ Cons: Still requires post-hoc step, duplicate model loads

**Option B: Integrate into data_preprocessing.py**
- ✅ Pros: Single preprocessing run, consistent with probes
- ❌ Cons: Mixing two codebases, more complex preprocessing script

**Recommendation**: **Option A** (keep separate) for now because:
- Logit lens already has integrated baseline correction
- Separate concerns (probes vs logit lens)
- Can run in parallel with probe preprocessing

**Implementation**:
Create master preprocessing script:
```python
# preprocess_complete.py
def preprocess_with_all_methods(data_path, output_path):
    # 1. Run probe-based preprocessing
    run_probe_preprocessing(data_path, temp_path, all_probe_types)

    # 2. Run logit lens (adds to existing pickle)
    run_logit_lens_preprocessing(temp_path, temp_path, all_layer_ranges)

    # 3. Run axis lens (adds to existing pickle)
    run_axis_lens_preprocessing(temp_path, output_path)

    # Done - single output file with everything!
```

---

### Phase 3: Eliminate Layer Range Post-Addition 🔧 **HIGH PRIORITY**

**Goal**: All logit lens layer ranges computed in single pass

**Current**: `add_logit_lens_layers.py` adds L20-30, L30-40 post-hoc

**Solution**: Update `add_logit_lens_to_existing.py` to compute all layer ranges at once:

```python
# add_logit_lens_to_existing.py

LAYER_RANGES = {
    'logit_lens_mean': list(range(40, 51)),  # L40-50
    'logit_lens_mean_l30_40': list(range(30, 41)),  # L30-40
    'logit_lens_mean_l20_30': list(range(20, 31)),  # L20-30
}

for probe_key, layers in LAYER_RANGES.items():
    # Compute scores for this layer range
    # Store in conv['probe_scores'][probe_key]
```

**Impact**: Eliminates `add_logit_lens_layers.py` and `fix_missing_logit_data.sh`

---

### Phase 4: Eliminate Copy/Remove Hacks 🧹 **MEDIUM PRIORITY**

**Goal**: Never need to copy or remove scores

**Solution**:
1. Preprocess all subsets from JSONL source (not from each other)
2. If need to regenerate, just re-run preprocessing (now fast since single pass)
3. Add validation to preprocessing to catch errors early

**Scripts to eliminate**:
- `copy_orthogonal_scores.py`
- `remove_orthogonal_scores.py`
- `recompute_logit_incremental.py`

---

## File Size Impact

### Current (Multiple Pickle Updates)
```
Initial: 100 MB
+ Orthogonal reg: 120 MB (rewrite)
+ Text probes: 140 MB (rewrite)
+ Logit lens: 180 MB (rewrite)
+ Axis lens: 200 MB (rewrite)
+ More layers: 250 MB (rewrite)

Total disk writes: 990 MB (1 GB)
```

### Proposed (Single Preprocessing)
```
Single run: 250 MB

Total disk writes: 250 MB
```

**Savings**: 4x reduction in disk I/O

---

## Time Impact

### Current (Sequential Post-Hoc)
```
1. Initial preprocessing:          ~30 min
2. Add orthogonal regularized:     ~25 min
3. Add text probes:                ~20 min
4. Add logit lens:                 ~30 min
5. Add axis lens:                  ~35 min
6. Add more logit layers:          ~30 min
7. Apply baseline correction:      ~2 min

Total: ~172 minutes (~3 hours)
```

### Proposed (Single Run)
```
1. Comprehensive preprocessing:    ~45 min
   (All probes + all logit lens layers + axis lens)

Total: ~45 minutes
```

**Savings**: 4x faster (127 minutes saved)

**Why faster?**:
- Model loaded once (not 6 times)
- Activations extracted once (not 6 times)
- Minimal disk I/O overhead
- Can parallelize some operations (e.g., multiple probe types on same activations)

---

## Migration Path

### Quick Wins (Can Do Now)

1. ✅ **DONE**: Integrate baseline correction into logit/axis lens
   - Eliminated: `apply_baseline_correction.py`
   - Time saved: 2 min per dataset

2. **TODO**: Add all layer ranges to logit lens script
   - Update: `add_logit_lens_to_existing.py`
   - Eliminate: `add_logit_lens_layers.py`, `fix_missing_logit_data.sh`
   - Time saved: 30 min per dataset

3. **TODO**: Add orthogonal_regularized to probe_configs.py
   - Update: `probe_configs.py`, `data_preprocessing.py`
   - Eliminate: `add_orthogonal_regularized_probes.py`
   - Time saved: 25 min per dataset

### Medium-Term (Next Sprint)

4. **TODO**: Create master preprocessing script
   - New: `preprocess_complete.py`
   - Calls: `data_preprocessing.py` + `add_logit_lens_to_existing.py` + `add_axis_lens_to_existing.py`
   - Eliminates need for manual steps

5. **TODO**: Add diverse_isolation to probe_configs
   - Eliminate: `add_diverse_isolation_probes.py`

### Long-Term (Future)

6. **TODO**: Fully integrate logit/axis lens into data_preprocessing.py
   - Unify codebases
   - True single-pass preprocessing

---

## Recommendations

### Immediate Actions (This Week)

1. ✅ **DONE**: Baseline correction integration
2. ⚡ **DO NOW**: Update logit lens script to compute all layer ranges at once
   - Modify: `add_logit_lens_to_existing.py`
   - Add loop over layer ranges
   - Test on one subset first

3. ⚡ **DO NOW**: Add orthogonal_regularized to main preprocessing
   - Add config to `probe_configs.py`
   - Test preprocessing with it included

### Next Steps (Next 2 Weeks)

4. Create `preprocess_complete.sh` wrapper script that runs:
   ```bash
   # 1. Standard probes
   python data_preprocessing.py --all-probes

   # 2. Logit/axis lens with correction
   python add_logit_lens_to_existing.py --all-layers
   python add_axis_lens_to_existing.py
   ```

5. Document new workflow in QUICK_START.md

6. Deprecate old post-hoc scripts with warning messages

### Future (Nice to Have)

7. Refactor to true single-pass preprocessing
8. Add automated validation/checksums
9. Implement incremental preprocessing (only process new conversations)

---

## Success Metrics

After integration:

✅ **Workflow simplicity**: 1 command vs 7 commands
✅ **Time efficiency**: 45 min vs 3 hours
✅ **Disk efficiency**: 250 MB writes vs 1 GB writes
✅ **Maintainability**: 1 script to maintain vs 13 scripts
✅ **Error reduction**: No manual steps = no forgotten steps

---

## Conclusion

The current pipeline works but has accumulated technical debt through post-hoc "add_*" scripts. By consolidating these into the initial preprocessing, we can:

- **Save 2-3 hours per dataset**
- **Reduce complexity** (1 script vs 13)
- **Eliminate manual steps** and associated errors
- **Improve maintainability**

**Priority order**:
1. 🔥 Multi-layer logit lens (high impact, easy)
2. 🔥 Orthogonal regularized in main preprocessing (high impact, medium difficulty)
3. 🟡 Master preprocessing wrapper (medium impact, easy)
4. 🟢 Eliminate copy/remove hacks (low impact, easy)
5. 🔵 Full integration (high impact, hard)

The first 3 can be done this week and will provide 80% of the benefits.
