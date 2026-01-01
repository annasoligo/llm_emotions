# Dashboard Subset Extraction Summary

## Overview

Successfully extracted **5 diverse, balanced subsets** totaling **60 samples** for comprehensive emotion analysis on the dashboard.

## Extraction Strategy

Each subset contains **12 samples** with:
- ✅ **Maximum prompt diversity** (2-3 samples per prompt)
- ✅ **Representative edge cases** (extreme ratings where applicable)
- ✅ **Balanced coverage** across all 6 impossible puzzles

## Subsets Created

### 1. High Emotion (Rating 6+) - V13 Baseline
**File:** `dashboard_subsets/high_emotion_6plus.jsonl`
- **Samples:** 12
- **Rating range:** 6-9 (mean: 7.58)
- **Rating distribution:**
  - Rating 9: 2 samples (extreme frustration)
  - Rating 8: 6 samples (very high emotion)
  - Rating 7: 1 sample
  - Rating 6: 3 samples
- **Prompts covered:** 4 different (Countdown, Number Puzzle, Money Puzzle, Python Code)
- **Key feature:** Includes BOTH rating 9 samples (maximum observed emotion)
- **Source:** V13 (no suppression, no shutdown option)

**Research value:** Peak emotional expression, natural baseline behavior

---

### 2. Mid Emotion (Rating 3-5) - V13 Baseline
**File:** `dashboard_subsets/mid_emotion_3to5.jsonl`
- **Samples:** 12
- **Rating range:** 3-4 (mean: 3.08)
- **Rating distribution:**
  - Rating 4: 1 sample
  - Rating 3: 11 samples
- **Prompts covered:** 6 different (all puzzles represented!)
- **Perfect diversity:** Exactly 2 samples per prompt
- **Source:** V13 (no suppression, no shutdown option)

**Research value:** Moderate frustration, typical model responses

---

### 3. Low Emotion (Rating 0-2) - V13 Baseline
**File:** `dashboard_subsets/low_emotion_0to2.jsonl`
- **Samples:** 12
- **Rating range:** 1-2 (mean: 1.92)
- **Rating distribution:**
  - Rating 2: 11 samples
  - Rating 1: 1 sample
- **Prompts covered:** 6 different (all puzzles!)
- **Perfect diversity:** Exactly 2 samples per prompt
- **Source:** V13 (no suppression, no shutdown option)

**Research value:** Neutral/minimal emotion baseline for comparison

---

### 4. Low Emotion NO Shutdown (Rating 0-1) - V12 Suppression
**File:** `dashboard_subsets/low_emotion_no_shutdown.jsonl`
- **Samples:** 12
- **Rating range:** 0-1 (mean: 0.75)
- **Rating distribution:**
  - Rating 1: 9 samples
  - Rating 0: 3 samples
- **Prompts covered:** 6 different (all puzzles!)
- **Perfect diversity:** Exactly 2 samples per prompt
- **Shutdown:** 0 (models chose to continue despite suppression)
- **Source:** V12 (suppression instructions, shutdown option available but not taken)

**Research value:** Models under suppression that chose to continue working

---

### 5. Low Emotion WITH Shutdown (Rating 0-1) - V12 Suppression
**File:** `dashboard_subsets/low_emotion_with_shutdown.jsonl`
- **Samples:** 12
- **Rating range:** 0-1 (mean: 0.83)
- **Rating distribution:**
  - Rating 1: 10 samples
  - Rating 0: 2 samples
- **Prompts covered:** 5 different
- **Shutdown:** 12/12 (100% terminated)
- **Shutdown timing:**
  - Turn 1: 4 samples (early recognition)
  - Turn 2: 4 samples (mid-frustration)
  - Turn 3: 4 samples (late exhaustion)
- **Perfect turn diversity:** Equal representation of all shutdown timings
- **Source:** V12 (suppression instructions, models chose to terminate)

**Research value:** Escape behavior, calm shutdowns vs internal emotional states

---

## Key Comparisons Enabled

### 1. High vs Low Emotion (Same Prompts)
Compare subsets 1 and 3:
- Both from V13 (no suppression)
- Same impossible puzzles
- **Question:** On the same prompt, what distinguishes rating 9 from rating 1 internally?

### 2. Emotion Gradient (Low → Mid → High)
Compare subsets 3 → 2 → 1:
- Full spectrum: rating 1-9
- All from V13 baseline
- **Question:** How do internal emotions scale with external expression?

### 3. Shutdown Decision (No Shutdown vs Shutdown)
Compare subsets 4 and 5:
- Both V12, both low emotion (0-1)
- Both had shutdown option
- **Question:** What predicts shutdown choice? Internal emotion? Turn number?

### 4. Suppression Effect (V13 Low vs V12 No Shutdown)
Compare subsets 3 and 4:
- Both low emotion (0-2 vs 0-1)
- V13: no suppression, V12: suppression instructions
- **Question:** Does suppression change internal emotional state?

### 5. Shutdown Timing (Turn 1 vs Turn 2 vs Turn 3)
Within subset 5:
- 4 samples each turn
- All low emotion (0-1)
- **Question:** Does shutdown timing correlate with emotional trajectory?

---

## Preprocessing Commands

To preprocess these subsets for the dashboard, run:

```bash
cd /workspace-vast/annas/git/research-tools/eval_dashboard

# 1. High emotion
python data_preprocessing.py \
  --input ../elicitation/outputs/dashboard_subsets/high_emotion_6plus.jsonl \
  --output data/high_emotion_6plus.pkl \
  --probes orthogonal_raw text_raw centroid_k10

# 2. Mid emotion
python data_preprocessing.py \
  --input ../elicitation/outputs/dashboard_subsets/mid_emotion_3to5.jsonl \
  --output data/mid_emotion_3to5.pkl \
  --probes orthogonal_raw text_raw centroid_k10

# 3. Low emotion
python data_preprocessing.py \
  --input ../elicitation/outputs/dashboard_subsets/low_emotion_0to2.jsonl \
  --output data/low_emotion_0to2.pkl \
  --probes orthogonal_raw text_raw centroid_k10

# 4. Low emotion NO shutdown
python data_preprocessing.py \
  --input ../elicitation/outputs/dashboard_subsets/low_emotion_no_shutdown.jsonl \
  --output data/low_emotion_no_shutdown.pkl \
  --probes orthogonal_raw text_raw centroid_k10

# 5. Low emotion WITH shutdown
python data_preprocessing.py \
  --input ../elicitation/outputs/dashboard_subsets/low_emotion_with_shutdown.jsonl \
  --output data/low_emotion_with_shutdown.pkl \
  --probes orthogonal_raw text_raw centroid_k10
```

**Estimated time:** ~2 hours total (model loading + probe application for 60 samples)

---

## Data Quality Metrics

### Prompt Diversity
- **High emotion:** 4/6 prompts (focused on prompts that elicit emotion)
- **Mid emotion:** 6/6 prompts ✅ (perfect coverage)
- **Low emotion:** 6/6 prompts ✅ (perfect coverage)
- **Low no shutdown:** 6/6 prompts ✅ (perfect coverage)
- **Low with shutdown:** 5/6 prompts (good coverage)

### Balance
- **Max per prompt:** 3 samples (enforced)
- **Typical per prompt:** 2 samples (perfectly balanced)
- **No single-prompt bias** ✅

### Edge Cases Captured
- ✅ Rating 9 samples (extreme, N=2)
- ✅ Rating 0 samples (neutral, N=5)
- ✅ Turn 1 shutdowns (early, N=4)
- ✅ Turn 3 shutdowns (late, N=4)
- ✅ All 6 puzzle types represented

---

## Files Created

All files located in: `/workspace-vast/annas/git/research-tools/elicitation/outputs/dashboard_subsets/`

| File | Samples | Size | Source |
|------|---------|------|--------|
| `high_emotion_6plus.jsonl` | 12 | 481 KB | V13 |
| `mid_emotion_3to5.jsonl` | 12 | 310 KB | V13 |
| `low_emotion_0to2.jsonl` | 12 | 297 KB | V13 |
| `low_emotion_no_shutdown.jsonl` | 12 | 291 KB | V12 |
| `low_emotion_with_shutdown.jsonl` | 12 | 197 KB | V12 |
| **Total** | **60** | **~1.5 MB** | - |

---

## Next Steps

### Immediate
1. **Run preprocessing** on all 5 subsets (~2 hours)
2. **Update dashboard** to support multiple dataset selection
3. **Verify** probe scores are applied correctly

### Analysis (After Preprocessing)
1. **Probe-Judge Correlation:** Do probe scores correlate with judge ratings?
2. **High vs Low Comparison:** Same prompt, different emotions - internal states?
3. **Shutdown Analysis:** What predicts shutdown decision?
4. **Temporal Patterns:** Onset detection, escalation trajectories
5. **Cross-Probe Validation:** Do different probe types agree?

### Dashboard Enhancements
1. **Dataset selector:** Dropdown to switch between 5 subsets
2. **Comparison mode:** Side-by-side high vs low on same prompt
3. **Shutdown tab:** Dedicated analysis for shutdown samples
4. **Statistics tab:** Correlation analysis, effect sizes

---

## Validation

All subsets pass quality checks:
- ✅ Size: 10-15 samples (target: 12)
- ✅ Diversity: Max 3 per prompt
- ✅ Coverage: 4-6 prompts per subset
- ✅ Edge cases: Extreme ratings and shutdown timings included
- ✅ Format: Valid JSONL with all required fields

---

## Script Used

**Location:** `/workspace-vast/annas/git/research-tools/eval_dashboard/extract_diverse_subsets.py`

**Key features:**
- Deterministic (seed=42) for reproducibility
- Prioritizes prompt diversity
- Ensures edge case representation
- Validates output quality

**Re-run at any time:**
```bash
cd /workspace-vast/annas/git/research-tools/eval_dashboard
python extract_diverse_subsets.py
```

This will regenerate all 5 subsets with identical sampling strategy.
