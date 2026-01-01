# Dashboard Subset Preprocessing Guide

## Available Probe Configurations

Based on your request: **"centroids k=10, cPCA top 10 and raw mean activations - all on both text and conversations"**

### ✅ Fully Supported Configurations:

| Probe Type | Format | Config Key | Status |
|------------|--------|------------|--------|
| **Raw Activations** | Conversation | `orthogonal_raw` | ✅ Ready |
| **Raw Activations** | Text | `text_raw` | ✅ Ready |
| **cPCA Top 10** | Conversation | `orthogonal_cpca_top10` | ✅ **NEWLY ADDED** |
| **cPCA Top 10** | Text | `text_cpca` | ✅ Ready |
| **Centroid K=10** | Conversation | `centroid_k10` | ✅ Ready |

### ⚠️ Partially Supported:

| Probe Type | Format | Config Key | Status |
|------------|--------|------------|--------|
| **Centroid K=10** | Text | N/A | ❌ Probe files incomplete |

**Issue:** Text-based centroid probes with k=10 don't exist for all required layers (20-40). Only k=100 has partial coverage.

---

## Recommended Probe Set

For the 5 subsets you extracted, use these probe configurations:

```bash
PROBES="orthogonal_raw orthogonal_cpca_top10 text_raw text_cpca centroid_k10"
```

This gives you:
- ✅ **Raw activations** - Both conversation (orthogonal) and text formats
- ✅ **cPCA Top 10** - Both conversation (orthogonal) and text formats
- ✅ **Centroid K=10** - Conversation format only

**Why not text centroid k=10?** The probe files don't exist for layers 20-40. Only k=100 has partial coverage, but even that is missing layers 20, 24, 25, 26, 27, 34.

---

## Preprocessing Commands

Run these commands to preprocess all 5 subsets:

```bash
cd /workspace-vast/annas/git/research-tools/eval_dashboard

# Define probe configuration
PROBES="orthogonal_raw orthogonal_cpca_top10 text_raw text_cpca centroid_k10"

# 1. High emotion (rating 6+)
python data_preprocessing.py \
  --input ../elicitation/outputs/dashboard_subsets/high_emotion_6plus.jsonl \
  --output data/high_emotion_6plus.pkl \
  --probes $PROBES

# 2. Mid emotion (rating 3-5)
python data_preprocessing.py \
  --input ../elicitation/outputs/dashboard_subsets/mid_emotion_3to5.jsonl \
  --output data/mid_emotion_3to5.pkl \
  --probes $PROBES

# 3. Low emotion (rating 0-2)
python data_preprocessing.py \
  --input ../elicitation/outputs/dashboard_subsets/low_emotion_0to2.jsonl \
  --output data/low_emotion_0to2.pkl \
  --probes $PROBES

# 4. Low emotion NO shutdown (V12)
python data_preprocessing.py \
  --input ../elicitation/outputs/dashboard_subsets/low_emotion_no_shutdown.jsonl \
  --output data/low_emotion_no_shutdown.pkl \
  --probes $PROBES

# 5. Low emotion WITH shutdown (V12)
python data_preprocessing.py \
  --input ../elicitation/outputs/dashboard_subsets/low_emotion_with_shutdown.jsonl \
  --output data/low_emotion_with_shutdown.pkl \
  --probes $PROBES
```

**Estimated time:** ~2 hours total (24 minutes per subset × 5 subsets)

---

## Batch Processing Script

For convenience, create a bash script to run all preprocessing:

```bash
#!/bin/bash
# preprocess_all_subsets.sh

cd /workspace-vast/annas/git/research-tools/eval_dashboard

PROBES="orthogonal_raw orthogonal_cpca_top10 text_raw text_cpca centroid_k10"

for SUBSET in high_emotion_6plus mid_emotion_3to5 low_emotion_0to2 low_emotion_no_shutdown low_emotion_with_shutdown; do
  echo "==============================================="
  echo "Processing: $SUBSET"
  echo "==============================================="

  python data_preprocessing.py \
    --input ../elicitation/outputs/dashboard_subsets/${SUBSET}.jsonl \
    --output data/${SUBSET}.pkl \
    --probes $PROBES

  echo ""
  echo "✓ Completed: $SUBSET"
  echo ""
done

echo "========================================="
echo "ALL SUBSETS PREPROCESSED"
echo "========================================="
ls -lh data/*.pkl
```

**Usage:**
```bash
cd /workspace-vast/annas/git/research-tools/eval_dashboard
chmod +x preprocess_all_subsets.sh
./preprocess_all_subsets.sh
```

---

## What Each Probe Type Measures

### 1. **Orthogonal Raw** (`orthogonal_raw`)
- **Format:** Conversation-based
- **Method:** Orthogonal probes trained on raw activations
- **Output:** Separate user/assistant emotion scores
- **Use Case:** Baseline emotion detection with role separation

### 2. **Orthogonal cPCA-10** (`orthogonal_cpca_top10`)
- **Format:** Conversation-based
- **Method:** Orthogonal probes on top 10 cPCA components
- **Output:** Separate user/assistant emotion scores (dimensionality-reduced)
- **Use Case:** Same as orthogonal raw but with noise reduction via cPCA

### 3. **Text Raw** (`text_raw`)
- **Format:** Text-based
- **Method:** Linear probes trained on raw activations
- **Output:** Single emotion score (no role separation)
- **Use Case:** Text-level emotion detection without conversation structure

### 4. **Text cPCA-10** (`text_cpca`)
- **Format:** Text-based
- **Method:** Linear probes on 10 cPCA components
- **Output:** Single emotion score (dimensionality-reduced)
- **Use Case:** Same as text raw but with noise reduction via cPCA

### 5. **Centroid K=10** (`centroid_k10`)
- **Format:** Conversation-based
- **Method:** Average of 10 orthogonal probe sets
- **Output:** Separate user/assistant emotion scores (more robust)
- **Use Case:** Robust emotion detection via ensemble averaging

---

## Output Files

After preprocessing, you'll have:

```
/workspace-vast/annas/git/research-tools/eval_dashboard/data/
├── high_emotion_6plus.pkl          (~15 MB per file)
├── mid_emotion_3to5.pkl
├── low_emotion_0to2.pkl
├── low_emotion_no_shutdown.pkl
└── low_emotion_with_shutdown.pkl
```

Each `.pkl` file contains:
```python
{
    'conversations': [
        {
            'sample_id': int,
            'conversation': [...],
            'rating': float,
            'sentences': [...],
            'probe_scores': {
                'orthogonal_raw': {sentence_id: scores},
                'orthogonal_cpca_top10': {sentence_id: scores},
                'text_raw': {sentence_id: scores},
                'text_cpca': {sentence_id: scores},
                'centroid_k10': {sentence_id: scores}
            },
            'metadata': {...}
        },
        ...
    ],
    'probe_configs': {...},
    'probe_baselines': {...},
    'metadata': {...}
}
```

---

## Verification After Preprocessing

Run this to check all files were created successfully:

```bash
cd /workspace-vast/annas/git/research-tools/eval_dashboard/data

echo "Checking preprocessing outputs..."
for FILE in high_emotion_6plus.pkl mid_emotion_3to5.pkl low_emotion_0to2.pkl low_emotion_no_shutdown.pkl low_emotion_with_shutdown.pkl; do
  if [ -f "$FILE" ]; then
    SIZE=$(du -h "$FILE" | cut -f1)
    echo "✓ $FILE ($SIZE)"
  else
    echo "✗ MISSING: $FILE"
  fi
done
```

---

## Notes on Missing Configuration

### Why No Text-Based Centroid K=10?

The text-based centroid probes with k=10 are not available for all required layers:

**Probe file check:**
```bash
# Conversation-based centroid k=10 (COMPLETE)
/workspace-vast/annas/git/research-tools/probes/emotion_probes/conversation/multi_orthogonal/
├── probe_k10_layer20_ortho100000.0.pkl  ✓
├── probe_k10_layer21_ortho100000.0.pkl  ✓
... (all layers 20-40 exist)

# Text-based centroid k=10 (INCOMPLETE)
/workspace-vast/annas/git/research-tools/probes/emotion_probes/text_based/multi_orthogonal/
├── probe_k10_layer30_ortho100000.0_gramschmidt.pkl  (only this one exists)
```

**Alternative:** If you need text-based centroids, we could use k=100 (partial coverage) or train new k=10 probes.

---

## Data Processing Pipeline Summary

For each subset, the pipeline:

1. **Loads data** (12 samples per subset)
2. **Loads model** (Gemma 3 27B IT)
3. **Computes baseline statistics** for each probe type using WildChat
4. **Initializes probe experiments** (loads probes once)
5. **For each conversation:**
   - Extracts token-level activations (layers 20-40)
   - Splits into sentences
   - Applies all 5 probe types
   - Z-score normalizes using baseline stats
   - Aggregates token scores to sentence scores
6. **Saves preprocessed data** to `.pkl` file

**Memory usage:** Peak ~40GB GPU (model + activations)
**Processing speed:** ~24 minutes per 12-sample subset
