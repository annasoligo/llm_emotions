# Probe Configuration Analysis

## Current Probe Configurations

### Available in probe_configs.py:

1. **orthogonal_raw** - Conversation-based orthogonal probes on raw activations ✓
2. **orthogonal_cpca_top20** - Conversation-based orthogonal probes on cPCA top 20 ✓
3. **text_raw** - Text-based linear probes on raw activations ✓
4. **text_cpca** - Text-based linear probes on cPCA (nc=10) ✓
5. **centroid_k10** - Conversation-based centroid probes (k=10) ✓
6. **centroid_k50** - Conversation-based centroid probes (k=50) ✓

---

## Requested Probe Configurations

User requested: **"I'd like to do it for centroids k=10, cPCA top 10 and raw mean activations - all on both text and conversations"**

### Required Configurations:

#### 1. Centroid K=10
- **Conversation format:** ✅ Already exists (`centroid_k10`)
- **Text format:** ❌ MISSING - Need to add `centroid_k10_text`

#### 2. cPCA Top 10
- **Conversation format:** ✅ Partially exists (`orthogonal_cpca_top20` uses top 20, not 10)
  - Need: `orthogonal_cpca_top10` for consistency
- **Text format:** ✅ Already exists (`text_cpca` uses nc=10)

#### 3. Raw Mean Activations
- **Conversation format:** ✅ Already exists (`orthogonal_raw`)
- **Text format:** ✅ Already exists (`text_raw`)

---

## Probe Files Available on Disk

### Conversation Format:
```
/workspace-vast/annas/git/research-tools/probes/emotion_probes/conversation/multi_orthogonal/
├── probe_k10_layer20_ortho100000.0.pkl  ✓ (centroid k=10)
├── probe_k10_layer21_ortho100000.0.pkl
└── ... (layers 20-40)
```

### Text Format:
```
/workspace-vast/annas/git/research-tools/probes/emotion_probes/text_based/multi_orthogonal/
├── best_k1_layer0_ortho100000.0.pkl
├── best_k1_layer30_ortho1000.0.pkl
└── ... (various k values and layers)
```

**Note:** Text-based centroid probes appear to exist but follow a different naming pattern (`best_k1_*`, etc.)

### cPCA Files:
```
Conversation-based cPCA:
/workspace-vast/annas/git/research-tools/outputs/dimensionality_reduction/cpca/conversation_based/global/google/google/gemma-3-27b-it_cpca.npz ✓

Text-based cPCA:
/workspace-vast/annas/git/research-tools/probes/results/cpca_tier_data_high_alpha.tmp/google/gemma-3-27b-it_cpca.npz ✓
```

---

## Configurations to Add

### 1. Text-Based Centroid K=10
```python
'centroid_k10_text': {
    'name': 'Centroid K=10 (Text)',
    'display_name': 'Centroid K=10 Text',
    'type': 'centroid',
    'probe_dir': RESEARCH_TOOLS / "probes/emotion_probes/text_based",
    'k_value': 10,
    'orthogonality_weight': 100000.0,
    'centroid_probe_format': 'text',
    'split_user_asst': True,  # Centroid returns orthogonal dict format
    'color_user': '#f39c12',
    'color_asst': '#d68910',
    'description': 'Text-based centroid probes averaging 10 orthogonal probe sets'
}
```

**Issue:** Need to verify text-based centroid probe file naming pattern. Current pattern appears to be `best_k{N}_layer{L}_ortho{weight}.pkl` rather than `probe_k{N}_layer{L}_ortho{weight}.pkl`.

### 2. Conversation-Based Orthogonal cPCA Top 10
```python
'orthogonal_cpca_top10': {
    'name': 'Orthogonal (User/Asst) - cPCA Top 10',
    'display_name': 'Orthogonal cPCA-10',
    'type': 'orthogonal',
    'probe_dir': PROBE_BASE / "conversation_based",
    'cpca_path': CPCA_BASE / "conversation_based/global/google/google/gemma-3-27b-it_cpca.npz",
    'orthogonality_weight': 1000.0,
    'orthogonal_representation': 'global_cpca_top10',
    'n_components': 10,
    'split_user_asst': True,
    'color_user': '#16a085',  # Dark teal for user
    'color_asst': '#c0392b',  # Dark red for assistant
    'description': 'Orthogonal probes on top 10 cPCA components'
}
```

---

## Summary: What We Already Have vs What We Need

### ✅ Already Covered:
1. **Conversation orthogonal raw** → `orthogonal_raw`
2. **Text linear raw** → `text_raw`
3. **Text linear cPCA-10** → `text_cpca`
4. **Conversation centroid k=10** → `centroid_k10`

### ✅ ADDED:
1. **Conversation orthogonal cPCA-10** → `orthogonal_cpca_top10` ✓ **COMPLETED**

### ❌ Cannot Support:
1. **Text centroid k=10** → Probe files don't exist for layers 20-40
   - Only 1 file exists: `probe_k10_layer30_ortho100000.0_gramschmidt.pkl`
   - Alternative k=100 has partial coverage (missing layers 20, 24-27, 34)
   - Would need to train new probes to support this configuration

---

## Recommended Preprocessing Commands

Once configurations are added, run preprocessing with:

```bash
cd /workspace-vast/annas/git/research-tools/eval_dashboard

# Probe configurations to use:
PROBES="orthogonal_raw orthogonal_cpca_top10 text_raw text_cpca centroid_k10 centroid_k10_text"

# For each subset:
for SUBSET in high_emotion_6plus mid_emotion_3to5 low_emotion_0to2 low_emotion_no_shutdown low_emotion_with_shutdown; do
  python data_preprocessing.py \
    --input ../elicitation/outputs/dashboard_subsets/${SUBSET}.jsonl \
    --output data/${SUBSET}.pkl \
    --probes $PROBES
done
```

**Estimated time:** ~2 hours total (24 minutes per subset × 5 subsets)

---

## Current Preprocessing Default

The default probes in `data_preprocessing.py` (line 427) are:
```python
default=['orthogonal_raw', 'text_raw', 'centroid_k10']
```

This covers:
- ✅ Conversation orthogonal raw
- ✅ Text linear raw
- ✅ Conversation centroid k=10

**Missing from default:**
- Conversation orthogonal cPCA-10
- Text linear cPCA-10
- Text centroid k=10
