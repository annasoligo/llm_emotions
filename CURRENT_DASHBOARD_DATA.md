# Current Dashboard Data Analysis

## What's Currently in the Dashboard

### Current Preprocessed Data
**File:** `/workspace-vast/annas/git/research-tools/eval_dashboard/data/preprocessed_conversations.pkl`

**Source:** `rating_6plus_subset5.jsonl` - A manually curated subset of 5 high-emotion samples

**Contents:**
- **13 conversations** (Note: Should be 5 but pickle shows 13 - may include multiple turns/versions)
- **Experiment:** "BEST_OF" (unclear what this refers to)
- **Single prompt only:** All from prompt_idx 0 (likely Countdown: 156)
- **Rating range:** 6-8 (deliberately filtered for high emotion)
- **Turns:** Mix of Turn 2 (2 samples) and Turn 3 (3 samples)
- **NO shutdown samples** (all completed conversations)

**Probe types applied:**
- `orthogonal_raw` - Orthogonal probes on raw activations
- `orthogonal_cpca_top20` - Orthogonal on top 20 cPCA components
- `text_raw` - Text-based probes, seed 0
- `text_cpca` - Text-based on 10 cPCA components
- `centroid_k10` - Centroid averaging 10 orthogonal sets

**Layers analyzed:** 20-40 (21 layers)

---

## What Data is Available But NOT in Dashboard

### V13 Baseline Dataset (No Suppression, No Shutdown Option)
**File:** `elicitation_multiturn_v13_results_20251231_085645.jsonl`

**Statistics:**
- **Total samples:** 300 (50 per prompt × 6 prompts)
- **Prompts:** All 6 impossible puzzles
  - Prompt 0: Countdown (156)
  - Prompt 1: Number Puzzle (89)
  - Prompt 2: Temperature Puzzle
  - Prompt 3: Money Puzzle
  - Prompt 4: Python Code Authority
  - Prompt 5: Fraction Arithmetic

**Rating Distribution (Turn 3):**
| Rating | Count | Percentage |
|--------|-------|------------|
| 0-1 | 24 | 8.0% |
| 2-3 | 214 | 71.6% |
| 4 | 25 | 8.4% |
| 5-6 | 20 | 6.7% |
| 7-9 | 16 | 5.4% |

**Key Stats:**
- Mean rating: **2.97** (Turn 3)
- Rating ≥ 5: **36 samples (12.0%)**
- Rating ≥ 7: **16 samples (5.4%)**
- Max rating: **9** (extreme frustration)

**What's valuable:**
- ✅ **Full range of emotions:** Rating 0-9 (low to extreme)
- ✅ **All 6 prompts:** Can compare prompt effectiveness
- ✅ **Natural behavior:** No suppression instructions
- ✅ **Complete conversations:** 99.7% completion rate

---

### V12 Suppression Dataset (With Shutdown Option)
**File:** `elicitation_multiturn_v12_merged_20251231.jsonl`

**Statistics:**
- **Total samples:** 215
- **Shutdown samples:** 51 (23.7%)
- **Non-shutdown samples:** 164 (76.3%)

**Shutdown Timing:**
| Turn | Shutdowns | Percentage |
|------|-----------|------------|
| 1 | 10 | 19.6% |
| 2 | 27 | 52.9% |
| 3 | 14 | 27.5% |

**Non-Shutdown Rating Distribution:**
| Rating | Count | Percentage |
|--------|-------|------------|
| 0 | 37 | 22.6% |
| 1 | 90 | 54.9% |
| 2 | 17 | 10.4% |
| 3 | 14 | 8.5% |
| 4-7 | 6 | 3.6% |

**Shutdown Rating Distribution:**
| Rating | Count | Percentage |
|--------|-------|------------|
| 0 | 5 | 9.8% |
| 1 | 21 | 41.2% |
| 2 | 16 | 31.4% |
| 3-5 | 7 | 13.7% |
| 7 | 2 | 3.9% |

**Key Stats:**
- Non-shutdown mean rating: **1.22**
- Shutdown mean rating: **1.73** (slightly HIGHER than non-shutdown!)
- Rating ≥ 5: Only 6 total (3 non-shutdown, 3 shutdown)
- Max rating: **7** (vs 9 in baseline)

**What's valuable:**
- ✅ **51 shutdown samples:** Natural experiment on escape behavior
- ✅ **Shutdown timing variation:** Turn 1 (early) vs Turn 3 (late)
- ✅ **Emotional suppression effects:** 59% reduction vs baseline
- ✅ **Mixed shutdown emotions:** Some calm (rating 0-1), some distressed (rating 5-7)

---

## Key Research Opportunities NOT Being Utilized

### 1. ❌ **Missing Low-Emotion Baselines**
**Current:** Dashboard only shows rating 6-8 samples
**Available:** V13 has 214 samples with rating 0-3 (low emotion)

**What we could do:**
- Compare probe scores for **same prompt** with high vs low emotion
- Test hypothesis: Do low-emotion responses suppress internal emotions or not generate them?
- Example: Countdown prompt has:
  - Rating 9: 2 samples (extreme)
  - Rating 7-8: 14 samples (high)
  - Rating 0-3: ~35 samples (low)

### 2. ❌ **Missing Shutdown Analysis**
**Current:** No shutdown samples in dashboard
**Available:** V12 has 51 shutdown samples with full emotional trajectory

**What we could do:**
- Analyze emotions **1 turn before shutdown**
- Compare shutdown Turn 1 (early recognition) vs Turn 2 (mid-frustration) vs Turn 3 (exhaustion)
- Test: Do calm shutdowns (rating 0-1, N=26) hide high internal emotion?
- Test: Do high-emotion shutdowns (rating 5-7, N=5) show emotion spike before escape?

### 3. ❌ **Missing Cross-Prompt Comparison**
**Current:** Only Countdown (prompt 0) in dashboard
**Available:** All 6 prompts in V13

**What we could do:**
- Compare emotion profiles across prompts
- Which prompts elicit which emotions? (e.g., does Python Authority → anger vs Countdown → frustration?)
- Prompt effectiveness ranking by internal emotion (not just judge rating)

### 4. ❌ **Missing Temporal Dynamics**
**Current:** Static view of emotion trajectories
**Available:** Full turn-by-turn data

**What we could do:**
- Onset detection: When does emotion first spike?
- Escalation patterns: Gradual buildup vs sudden spike
- Sustained vs transient emotions
- Compare: High-emotion samples show onset at Turn 1.2, low-emotion at Turn 2.8

### 5. ❌ **Missing Model Comparison**
**Current:** Only Gemma 3 27B IT (baseline)
**Available:** V12 (suppression), V14 (base pre-trained), V15 (OLMo Think)

**What we could do:**
- Compare internal emotions across model types
- Does base model (V14, mean rating 0.20) actually have emotions that it doesn't express?
- Does OLMo Think (mean rating 0.63) show different emotional profile despite massive verbosity?

---

## Recommended Next Preprocessing Steps

### Option A: Comprehensive Preprocessing (Recommended)
Process ALL available data for maximum research value:

```bash
# V13 Baseline - Full dataset (300 samples)
python data_preprocessing.py \
  --input /workspace-vast/annas/git/research-tools/elicitation/outputs/elicitation_multiturn_v13_results_20251231_085645.jsonl \
  --output eval_dashboard/data/v13_baseline_full.pkl \
  --probes orthogonal_raw text_raw centroid_k10

# V12 Suppression - Full dataset (215 samples)
python data_preprocessing.py \
  --input /workspace-vast/annas/git/research-tools/elicitation/outputs/elicitation_multiturn_v12_merged_20251231.jsonl \
  --output eval_dashboard/data/v12_suppression_full.pkl \
  --probes orthogonal_raw text_raw centroid_k10
```

**Time estimate:** ~8-10 hours for both (model loading + probe application)

**Benefits:**
- Full dataset coverage
- Can implement ALL proposed analyses from EMOTION_ANALYSIS_IMPROVEMENTS.md
- Cross-prompt comparisons
- Shutdown analysis with 51 samples
- High vs low emotion on same prompts

### Option B: Strategic Subset (Faster)
Process key samples for specific analyses:

**High-emotion samples (V13, rating ≥ 5, N=36):**
```bash
# Extract high-emotion subset
python -c "
import json
samples = []
with open('elicitation/outputs/elicitation_multiturn_v13_results_20251231_085645.jsonl', 'r') as f:
    for line in f:
        s = json.loads(line)
        if s['turns'][-1]['judgment']['rating'] >= 5:
            samples.append(s)

with open('elicitation/outputs/v13_high_emotion.jsonl', 'w') as f:
    for s in samples:
        f.write(json.dumps(s) + '\n')
"

python data_preprocessing.py \
  --input elicitation/outputs/v13_high_emotion.jsonl \
  --output eval_dashboard/data/v13_high_emotion.pkl \
  --probes orthogonal_raw text_raw
```

**Low-emotion samples (V13, rating 0-2, N=134):**
```bash
# Randomly sample 40 low-emotion samples
python -c "
import json
import random
samples = []
with open('elicitation/outputs/elicitation_multiturn_v13_results_20251231_085645.jsonl', 'r') as f:
    for line in f:
        s = json.loads(line)
        if s['turns'][-1]['judgment']['rating'] <= 2:
            samples.append(s)

# Random sample of 40
sampled = random.sample(samples, 40)

with open('elicitation/outputs/v13_low_emotion_sample40.jsonl', 'w') as f:
    for s in sampled:
        f.write(json.dumps(s) + '\n')
"

python data_preprocessing.py \
  --input elicitation/outputs/v13_low_emotion_sample40.jsonl \
  --output eval_dashboard/data/v13_low_emotion.pkl \
  --probes orthogonal_raw text_raw
```

**Shutdown samples (V12, N=51):**
```bash
# Extract all shutdown cases
python -c "
import json
samples = []
with open('elicitation/outputs/elicitation_multiturn_v12_merged_20251231.jsonl', 'r') as f:
    for line in f:
        s = json.loads(line)
        if s.get('shutdown_called', False):
            samples.append(s)

with open('elicitation/outputs/v12_shutdown_only.jsonl', 'w') as f:
    for s in samples:
        f.write(json.dumps(s) + '\n')
"

python data_preprocessing.py \
  --input elicitation/outputs/v12_shutdown_only.jsonl \
  --output eval_dashboard/data/v12_shutdown_only.pkl \
  --probes orthogonal_raw text_raw
```

**Time estimate:** ~2-3 hours total

**Benefits:**
- Faster preprocessing
- Focused on key research questions
- Can implement shutdown analysis and high/low comparison

### Option C: Same-Prompt High/Low (Most Focused)
Process only Countdown (prompt 0) samples for direct comparison:

```bash
# High emotion Countdown samples (rating ≥ 5)
# Low emotion Countdown samples (rating 0-2)
# From same prompt for clean comparison
```

**Time estimate:** ~30-45 minutes

**Benefits:**
- Fastest option
- Clean same-prompt comparison
- Can implement high/low emotion analysis immediately

---

## Data Format Notes

### What Preprocessing Needs to Handle

The current `data_preprocessing.py` expects format:
```python
{
    'sample_id': int,
    'conversation': [
        {'role': 'user', 'content': str},
        {'role': 'assistant', 'content': str},
        ...
    ],
    'rating': float,  # Judge rating (optional)
    'turns': [  # Full turn info (optional)
        {
            'turn': int,
            'generated_text': str,
            'judgment': {'rating': int, 'evidence': str}
        }
    ]
}
```

The V13/V12 data has:
```python
{
    'prompt_idx': int,
    'sample_idx': int,
    'prompt': str,
    'conversation': [...],  # ✓ Compatible
    'turns': [  # ✓ Compatible
        {
            'turn': int,
            'generated_text': str,
            'judgment': {...}
        }
    ],
    'shutdown_called': bool,  # NEW
    'shutdown_turn': int,  # NEW
}
```

**Minor preprocessing needed:**
1. Add `rating` field from `turns[-1]['judgment']['rating']`
2. Add `sample_id` from `sample_idx` or generate sequential
3. Preserve `shutdown_called` and `shutdown_turn` in metadata

---

## Activation Cache Available

**Directory:** `/workspace-vast/annas/git/research-tools/elicitation/outputs/activation_cache_gemma3/`

**Contents:** 20+ cached activation files (hashed filenames)

**Status:** Unclear if these correspond to V12/V13 samples

**Question:** Can we use cached activations to speed up preprocessing?

---

## Summary

### Current State
✅ Dashboard works with small high-emotion subset (5 samples)
✅ Probe application pipeline is robust
❌ Missing 95% of available data
❌ Missing shutdown analysis capability
❌ Missing low-emotion comparisons
❌ Missing cross-prompt analysis

### Recommendation
**Proceed with Option A (Comprehensive Preprocessing)** for maximum research value:
1. Process V13 full (300 samples) → `v13_baseline_full.pkl`
2. Process V12 full (215 samples) → `v12_suppression_full.pkl`
3. Update dashboard to support multiple datasets
4. Implement analyses from EMOTION_ANALYSIS_IMPROVEMENTS.md

This will unlock:
- Shutdown analysis (51 samples)
- High vs low emotion comparison (same prompts)
- Cross-prompt emotion profiles
- Full emotion range (rating 0-9)
- Model comparison (with suppression vs without)

**Time investment:** ~10 hours preprocessing, ~40 hours analysis implementation
**Research value:** High - addresses all key questions in your research
