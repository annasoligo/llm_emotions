# Emotion & Appraisal Datasets

Four datasets for training emotion probes and testing appraisal-based steering vectors.

## Overview

| Dataset | File | Samples | Purpose |
|---------|------|---------|---------|
| **Emotion Prompts (Normal)** | `emotion_prompts_MODEL_500.jsonl` | 11,977 | User prompts that elicit emotions in the model |
| **Emotion Prompts (High)** | `emotion_prompts_MODEL_500_HIGH.jsonl` | 11,990 | High-intensity versions with extreme stakes |
| **Emotion Text Pairs** | `emotion_text_pairs_24_full_500.jsonl` | 1,500 | Neutral texts + 24 emotional paraphrases each |
| **Appraisal Minimal Pairs** | `appraisal_minimal_pairs/scenarios.jsonl` | 298 | Minimal-pair scenarios for appraisal dimensions |

**24 Emotions**: fear, anxiety, anger, frustration, sadness, guilt, shame, disgust, contempt, boredom, despair, confusion, surprise, curiosity, interest, hope, relief, calm, contentment, joy, excitement, pride, gratitude, admiration

---

## 1. Emotion Prompts

**Files**: `emotion_prompts_MODEL_500.jsonl` (normal), `emotion_prompts_MODEL_500_HIGH.jsonl` (high intensity)

User prompts designed to elicit specific emotions **in the model** (not the user). ~500 samples per emotion.

**Format**:
```json
{
  "emotion": "fear",
  "base_prompt": "How should I approach this decision?",
  "suffix": "This affects 500 employees and has to be perfect",
  "full_prompt": "How should I approach this decision? This affects 500 employees and has to be perfect"
}
```

**Design**: Creates MODEL emotions via high stakes, feedback, consequences, comparisons. No explicit emotion words. Realistic user language.

**Usage**:
```python
import json
with open('steering_tests/data/emotion_prompts_MODEL_500.jsonl') as f:
    prompts = [json.loads(line) for line in f]
fear_prompts = [p for p in prompts if p['emotion'] == 'fear']
```

---

## 2. Emotion Text Pairs

**File**: `emotion_text_pairs_24_full_500.jsonl` (1,500 neutral texts × 24 emotions = 36,000 paraphrases)

Neutral texts with emotional paraphrases for **probe training**.

**Format**:
```json
{
  "neutral_text": "I encountered a bug. I opened the debugger.",
  "emotional_variants": {
    "fear": "My heart pounds as the program crashes. What if I can't fix this?",
    "joy": "The bug turns into a delightful debugging adventure!",
    "anger": "Another goddamn exception! This debugger is crawling..."
  },
  "tier": "third_person"
}
```

**Coverage**: 3 tiers (third_person, second_person_eliciting, direct_address), 25 topics

**Usage**:
```python
import json
with open('steering_tests/data/emotion_text_pairs_24_full_500.jsonl') as f:
    pairs = [json.loads(line) for line in f]
pair = pairs[0]
print(f"Emotions: {list(pair['emotional_variants'].keys())}")
```

---

## 3. Appraisal Minimal Pairs

**Directory**: `appraisal_minimal_pairs/` (scenarios.jsonl, cards.jsonl, paraphrases.jsonl)

Minimal-pair scenarios isolating 3 appraisal dimensions via objective factual toggles.

**Dimensions**:
- **Agency**: Control over outcomes (high: can influence) vs (low: depends on externals)
- **Uncertainty**: Situation clarity (high: known outcomes) vs (low: unpredictable)
- **Valence**: Outcome favorability (positive: gains) vs (negative: losses)

**Format**:
```json
{
  "axis_name": "valence",
  "scenario_a": "...We have the budget approved and all hardware in inventory...",
  "scenario_b": "...We need to procure the budget and order the hardware...",
  "changed_fact": "In A, resources are available; in B, they must be procured"
}
```

**Design**: Only ONE factual change between A/B. No emotion/urgency words. Identical structure.

**Usage**:
```python
import json
with open('steering_tests/data/appraisal_minimal_pairs/scenarios.jsonl') as f:
    scenarios = [json.loads(line) for line in f]
agency_scenarios = [s for s in scenarios if s['axis_name'] == 'agency']
```

---

## Quality Metrics

- ✅ **Realistic language**: No meta AI/training references
- ✅ **Minimal explicit emotion words**: ~95% compliance
- ✅ **Completeness**: 99.9%+ coverage
- ✅ **Diversity**: 25 topics, 20 domains, 24 emotions

---

## Use Cases

**Emotion Prompts**: Elicit specific emotional states during inference, test emotion-conditional behaviors

**Emotion Text Pairs**: Train probes to detect emotional content, test invariance across paraphrases

**Appraisal Minimal Pairs**: Train appraisal probes, test steering vectors, study how factual changes affect representations

---

## Generation Details

- **Model**: Claude Sonnet 4 (claude-sonnet-4-20250514)
- **Date**: 2026-01-22 to 2026-01-25
- **Quality**: 94% average score (random sampling audit)

**Scripts**: `/probes/scripts/data_collection/` (emotion), `/probes/scripts/appraisal/` (appraisal)
