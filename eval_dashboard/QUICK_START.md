# Quick Start: Dashboard Preprocessing

## TL;DR

Run this to preprocess all 5 subsets with all available probe types:

```bash
cd /workspace-vast/annas/git/research-tools/eval_dashboard
./preprocess_all_subsets.sh
```

**Time:** ~2 hours | **Output:** 5 preprocessed `.pkl` files in `data/`

---

## What Gets Applied

### ✅ 5 Probe Configurations:

1. **orthogonal_raw** - Conversation-based orthogonal probes on raw activations
2. **orthogonal_cpca_top10** - Conversation-based orthogonal probes on cPCA top 10 ✨ *NEW*
3. **text_raw** - Text-based linear probes on raw activations
4. **text_cpca** - Text-based linear probes on cPCA top 10
5. **centroid_k10** - Conversation-based centroid probes (k=10)

### ✅ To 5 Subsets:

1. **high_emotion_6plus** - 12 samples, rating 6-9 (V13 baseline)
2. **mid_emotion_3to5** - 12 samples, rating 3-5 (V13 baseline)
3. **low_emotion_0to2** - 12 samples, rating 0-2 (V13 baseline)
4. **low_emotion_no_shutdown** - 12 samples, rating 0-1, no termination (V12 suppression)
5. **low_emotion_with_shutdown** - 12 samples, rating 0-1, terminated (V12 suppression)

---

## What You Requested vs What's Available

| Request | Conversation Format | Text Format |
|---------|---------------------|-------------|
| **Raw activations** | ✅ orthogonal_raw | ✅ text_raw |
| **cPCA Top 10** | ✅ orthogonal_cpca_top10 *NEW* | ✅ text_cpca |
| **Centroid K=10** | ✅ centroid_k10 | ❌ Files missing |

**Result:** 5 out of 6 configurations supported.

---

## Manual Commands

If you prefer to run individually:

```bash
cd /workspace-vast/annas/git/research-tools/eval_dashboard

PROBES="orthogonal_raw orthogonal_cpca_top10 text_raw text_cpca centroid_k10"

# High emotion
python data_preprocessing.py \
  --input ../elicitation/outputs/dashboard_subsets/high_emotion_6plus.jsonl \
  --output data/high_emotion_6plus.pkl \
  --probes $PROBES

# Mid emotion
python data_preprocessing.py \
  --input ../elicitation/outputs/dashboard_subsets/mid_emotion_3to5.jsonl \
  --output data/mid_emotion_3to5.pkl \
  --probes $PROBES

# Low emotion
python data_preprocessing.py \
  --input ../elicitation/outputs/dashboard_subsets/low_emotion_0to2.jsonl \
  --output data/low_emotion_0to2.pkl \
  --probes $PROBES

# Low emotion NO shutdown
python data_preprocessing.py \
  --input ../elicitation/outputs/dashboard_subsets/low_emotion_no_shutdown.jsonl \
  --output data/low_emotion_no_shutdown.pkl \
  --probes $PROBES

# Low emotion WITH shutdown
python data_preprocessing.py \
  --input ../elicitation/outputs/dashboard_subsets/low_emotion_with_shutdown.jsonl \
  --output data/low_emotion_with_shutdown.pkl \
  --probes $PROBES
```

---

## Verification

After preprocessing completes, verify output:

```bash
cd /workspace-vast/annas/git/research-tools/eval_dashboard/data
ls -lh *.pkl
```

Expected:
```
-rw-r--r-- 1 annas annas 15M Jan  1 12:00 high_emotion_6plus.pkl
-rw-r--r-- 1 annas annas 15M Jan  1 12:30 mid_emotion_3to5.pkl
-rw-r--r-- 1 annas annas 15M Jan  1 13:00 low_emotion_0to2.pkl
-rw-r--r-- 1 annas annas 15M Jan  1 13:30 low_emotion_no_shutdown.pkl
-rw-r--r-- 1 annas annas 10M Jan  1 14:00 low_emotion_with_shutdown.pkl
```

---

## Documentation

- **[STATISTICS_EXTRACTION_SUMMARY.md](STATISTICS_EXTRACTION_SUMMARY.md)** - Complete summary
- **[PREPROCESSING_GUIDE.md](PREPROCESSING_GUIDE.md)** - Detailed guide
- **[PROBE_CONFIGURATION_ANALYSIS.md](PROBE_CONFIGURATION_ANALYSIS.md)** - Probe file analysis

---

## What's New

✨ **Added `orthogonal_cpca_top10` configuration** to [probe_configs.py](probe_configs.py)

This enables conversation-based emotion analysis on top 10 cPCA components, matching the text-based `text_cpca` configuration.
