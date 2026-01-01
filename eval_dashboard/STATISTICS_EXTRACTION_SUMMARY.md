# Statistics Extraction Summary

## Request Analysis

**User request:** "I'd like to do it for centroids k=10, cPCA top 10 and raw mean activations - all on both text and conversations"

This breaks down into 6 probe configurations:

| # | Probe Type | Format | Configuration Key | Status |
|---|------------|--------|-------------------|--------|
| 1 | Raw activations | Conversation | `orthogonal_raw` | ✅ Exists |
| 2 | Raw activations | Text | `text_raw` | ✅ Exists |
| 3 | cPCA Top 10 | Conversation | `orthogonal_cpca_top10` | ✅ **ADDED** |
| 4 | cPCA Top 10 | Text | `text_cpca` | ✅ Exists |
| 5 | Centroid K=10 | Conversation | `centroid_k10` | ✅ Exists |
| 6 | Centroid K=10 | Text | N/A | ❌ Not available |

**Result:** 5 out of 6 configurations are supported.

---

## What Was Done

### 1. Configuration Added
Added new probe configuration to [probe_configs.py](probe_configs.py):

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
    'color_user': '#16a085',
    'color_asst': '#c0392b',
    'description': 'Orthogonal probes on top 10 cPCA components'
}
```

### 2. Documentation Created
- **[PROBE_CONFIGURATION_ANALYSIS.md](PROBE_CONFIGURATION_ANALYSIS.md)** - Detailed analysis of available probe files and configurations
- **[PREPROCESSING_GUIDE.md](PREPROCESSING_GUIDE.md)** - Complete guide for preprocessing all 5 subsets
- **[preprocess_all_subsets.sh](preprocess_all_subsets.sh)** - Batch script to run preprocessing

### 3. Probe File Investigation
Verified availability of probe files on disk:

**Conversation-based:**
- ✅ Raw orthogonal probes: Full coverage (layers 20-40)
- ✅ Centroid k=10: Full coverage (layers 20-40)
- ✅ cPCA transformations: Available

**Text-based:**
- ✅ Raw linear probes: Full coverage (layers 20-40)
- ✅ cPCA linear probes (nc=10): Full coverage (layers 20-40)
- ❌ Centroid k=10: Only 1 file (layer 30)
- ⚠️ Centroid k=100: Partial coverage (missing layers 20, 24-27, 34)

---

## Recommended Probe Set

For preprocessing the 5 extracted subsets, use:

```bash
PROBES="orthogonal_raw orthogonal_cpca_top10 text_raw text_cpca centroid_k10"
```

This provides:
1. **Raw activations** - Both formats ✓
2. **cPCA Top 10** - Both formats ✓
3. **Centroid K=10** - Conversation only (text not available)

---

## Quick Start: Run Preprocessing

Execute preprocessing for all 5 subsets:

```bash
cd /workspace-vast/annas/git/research-tools/eval_dashboard
./preprocess_all_subsets.sh
```

Or run individually:

```bash
cd /workspace-vast/annas/git/research-tools/eval_dashboard

PROBES="orthogonal_raw orthogonal_cpca_top10 text_raw text_cpca centroid_k10"

# Process each subset
python data_preprocessing.py \
  --input ../elicitation/outputs/dashboard_subsets/high_emotion_6plus.jsonl \
  --output data/high_emotion_6plus.pkl \
  --probes $PROBES

# ... (repeat for other 4 subsets)
```

**Time estimate:** ~2 hours for all 5 subsets

---

## Output Structure

Each preprocessed `.pkl` file contains:

```python
{
    'conversations': [
        {
            'sample_id': int,
            'conversation': List[Dict],  # Full conversation turns
            'rating': float,             # Judge rating
            'sentences': List[Dict],     # Sentence segmentation
            'probe_scores': {
                'orthogonal_raw': {sentence_id: np.ndarray},         # Conversation raw
                'orthogonal_cpca_top10': {sentence_id: np.ndarray},  # Conversation cPCA-10
                'text_raw': {sentence_id: np.ndarray},               # Text raw
                'text_cpca': {sentence_id: np.ndarray},              # Text cPCA-10
                'centroid_k10': {sentence_id: np.ndarray}            # Conversation centroid
            },
            'metadata': {
                'num_sentences': int,
                'num_tokens': int,
                'onset_sentence_id': int,
                'judge_evidence': str,
                'shutdown_called': bool  # For V12 subsets
            }
        },
        ...  # 12 conversations per subset
    ],
    'probe_configs': Dict,      # Configuration used for each probe
    'probe_baselines': Dict,    # Z-score normalization statistics
    'metadata': Dict            # Layers, emotions, etc.
}
```

---

## Probe Score Format

### Orthogonal Probes (User/Assistant Separation)
Scores stored as nested dict:
```python
{
    'user': np.ndarray([anger, disgust, fear, happiness, sadness, surprise]),
    'assistant': np.ndarray([anger, disgust, fear, happiness, sadness, surprise])
}
```

### Linear/Text Probes (No Separation)
Scores stored as single array:
```python
np.ndarray([anger, disgust, fear, happiness, sadness, surprise])
```

### Normalization
All scores are z-score normalized using WildChat baseline statistics:
```
z-score = (raw_score - baseline_mean) / (baseline_std + epsilon)
```

---

## Analysis Coverage

### What You Can Analyze

✅ **Raw vs cPCA Comparison:**
- Conversation: `orthogonal_raw` vs `orthogonal_cpca_top10`
- Text: `text_raw` vs `text_cpca`
- Question: Does dimensionality reduction improve signal quality?

✅ **Conversation vs Text Comparison:**
- Raw: `orthogonal_raw` vs `text_raw`
- cPCA-10: `orthogonal_cpca_top10` vs `text_cpca`
- Question: How does conversation structure affect emotion detection?

✅ **Centroid Robustness:**
- `centroid_k10` (ensemble of 10 probes)
- Question: Is ensemble averaging more robust than single probes?

✅ **User vs Assistant Emotions:**
- Available for: `orthogonal_raw`, `orthogonal_cpca_top10`, `centroid_k10`
- Question: Do user and assistant emotions differ?

### What You Cannot Analyze (Yet)

❌ **Text-based Centroid K=10:**
- Probe files don't exist for required layers
- Would need to train new probes

---

## Next Steps

### 1. Run Preprocessing (~2 hours)
```bash
cd /workspace-vast/annas/git/research-tools/eval_dashboard
./preprocess_all_subsets.sh
```

### 2. Verify Output
```bash
cd /workspace-vast/annas/git/research-tools/eval_dashboard/data
ls -lh *.pkl
```

Expected files:
- `high_emotion_6plus.pkl` (~15 MB)
- `mid_emotion_3to5.pkl` (~15 MB)
- `low_emotion_0to2.pkl` (~15 MB)
- `low_emotion_no_shutdown.pkl` (~15 MB)
- `low_emotion_with_shutdown.pkl` (~10 MB, shorter conversations)

### 3. Update Dashboard
Modify `app.py` to:
- Support dataset selection (dropdown for 5 subsets)
- Display all 5 probe types
- Add comparison views (high vs low emotion on same prompt)
- Add shutdown analysis tab

### 4. Run Analyses
Implement analyses from [EMOTION_ANALYSIS_IMPROVEMENTS.md](../EMOTION_ANALYSIS_IMPROVEMENTS.md):
- Probe-judge correlation
- Temporal onset detection
- Shutdown prediction analysis
- High vs low emotion comparison

---

## Files Created

| File | Purpose |
|------|---------|
| [PROBE_CONFIGURATION_ANALYSIS.md](PROBE_CONFIGURATION_ANALYSIS.md) | Detailed probe configuration analysis |
| [PREPROCESSING_GUIDE.md](PREPROCESSING_GUIDE.md) | Complete preprocessing guide |
| [STATISTICS_EXTRACTION_SUMMARY.md](STATISTICS_EXTRACTION_SUMMARY.md) | This document |
| [preprocess_all_subsets.sh](preprocess_all_subsets.sh) | Batch preprocessing script |

## Configuration Changes

| File | Change |
|------|--------|
| [probe_configs.py](probe_configs.py) | Added `orthogonal_cpca_top10` configuration |

---

## Summary

✅ **5 out of 6 requested configurations are supported**
✅ **Added missing conversation-based cPCA-10 configuration**
✅ **Created preprocessing pipeline for all 5 subsets**
✅ **Documentation and batch scripts ready**
❌ **Text-based centroid k=10 not available (probe files missing)**

**Ready to preprocess!** Run `./preprocess_all_subsets.sh` to begin.
