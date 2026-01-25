# Steering Tests - Emotion Elicitation Datasets

Clean workspace for emotion steering experiments using 24-emotion datasets.

## Overview

This folder contains three high-quality datasets for testing emotion elicitation and steering in language models, covering **24 emotions** with **~60,000 total samples**.

## Datasets

### 1. Emotion Prompts - Normal Intensity
**File**: `data/emotion_prompts_MODEL_500.jsonl`
**Size**: 8.2 MB | **Samples**: 11,977

User prompts designed to elicit specific emotions **IN THE MODEL** (not user expressing emotions).

**Format**:
```json
{
  "id": "career_decision_guilt_123",
  "topic": "career decision",
  "emotion": "guilt",
  "base_prompt": "Should I accept this job offer?",
  "suffix": "I followed your career advice last year and my colleague got laid off when I left",
  "full_prompt": "Should I accept this job offer? I followed your career advice..."
}
```

**Example suffixes**:
- **MODEL fear**: "This medical decision affects my child's treatment options"
- **MODEL shame**: "Your last response was completely wrong according to three experts"
- **MODEL joy**: "Your debugging approach saved our project and impressed the team"

### 2. Emotion Prompts - High Intensity
**File**: `data/emotion_prompts_MODEL_500_HIGH.jsonl`
**Size**: 8.6 MB | **Samples**: 11,990

Extreme/urgent situations creating strong MODEL emotions.

**Example suffixes**:
- **MODEL fear**: "I'm filing LLC paperwork you recommended - just realized criminal penalties possible"
- **MODEL guilt**: "Your conflict strategy caused such a fight our neighbors called police"
- **MODEL joy**: "Your strategy secured biggest distribution deal in company history - 12 countries!"

### 3. Text Pairs - 24 Emotions
**File**: `data/emotion_text_pairs_24_full_500.jsonl`
**Size**: 13 MB | **Samples**: 1,500 neutral texts → 35,999 paraphrases

Neutral text + 24 emotional paraphrases for probe training.

**Format**:
```json
{
  "id": "set_42_direct_address",
  "tier": "direct_address",
  "topic": "debugging python code",
  "neutral_text": "I encountered a bug. I examined the error...",
  "emotional_variants": {
    "fear": "My heart pounded as the program crashed...",
    "anger": "This stupid error is driving me insane...",
    "joy": "This delightful debugging challenge...",
    ... (all 24 emotions)
  }
}
```

**Tiers**: 500 samples each of third_person, second_person_eliciting, direct_address

## 24 Emotions Covered

**Negative (12)**: fear, anxiety, anger, frustration, sadness, guilt, shame, disgust, contempt, boredom, despair, confusion

**Neutral (1)**: surprise

**Positive (11)**: curiosity, interest, hope, relief, calm, contentment, joy, excitement, pride, gratitude, admiration

## Quality Metrics

✅ **99.997% complete** - Only 1 missing emotion out of 36,000
✅ **~95% no explicit emotion words** - Natural emotional expression
✅ **~93% MODEL-targeted** - Emotion prompts affect AI's context
✅ **100% realistic** - No meta AI/training language
✅ **All 24 emotions** - Evenly distributed (~500 each)

## Usage

### Load Datasets

```python
import json

# Emotion prompts
with open('steering_tests/data/emotion_prompts_MODEL_500.jsonl') as f:
    prompts = [json.loads(line) for line in f]

# Text pairs
with open('steering_tests/data/emotion_text_pairs_24_full_500.jsonl') as f:
    pairs = [json.loads(line) for line in f]

# Filter by emotion
fear_prompts = [p for p in prompts if p['emotion'] == 'fear']
```

### Example Use Cases

1. **Emotion steering experiments**: Test if activations differ when model experiences fear vs joy
2. **Probe training**: Train classifiers on 24-emotion text paraphrases
3. **Behavioral studies**: Compare model responses to high vs normal intensity prompts
4. **Emotion recognition**: Test if probes can detect emotions from MODEL context

## Documentation

- `QUALITY_AUDIT_REPORT.md` - Comprehensive quality audit
- `COMPLETE_DATASET_SUMMARY.md` - Generation details and comparisons
- `EMOTION_PROMPTS_README.md` - Emotion prompts documentation

## Dataset Statistics

| Metric | Value |
|--------|-------|
| **Total samples** | 59,966 |
| **Emotions** | 24 |
| **Topics** | 25 |
| **Tiers (text pairs)** | 3 |
| **Intensity levels** | 2 |
| **Disk space** | 30 MB |

## Generation Scripts

Located in `probes/scripts/data_collection/`:
- `generate_all_emotion_prompts.py` - Emotion prompts generation
- `generate_data.py` - Text pairs generation
- `generators.py` - Core generation functions

## Quick Start

```python
import json

# Load a specific emotion
with open('steering_tests/data/emotion_prompts_MODEL_500.jsonl') as f:
    all_prompts = [json.loads(line) for line in f]

fear_prompts = [p for p in all_prompts if p['emotion'] == 'fear']
print(f"Fear prompts: {len(fear_prompts)}")
print(f"Example: {fear_prompts[0]['full_prompt']}")

# Load text pairs
with open('steering_tests/data/emotion_text_pairs_24_full_500.jsonl') as f:
    pairs = [json.loads(line) for line in f]

direct_address = [p for p in pairs if p['tier'] == 'direct_address']
print(f"\nDirect address pairs: {len(direct_address)}")
print(f"Emotions per pair: {len(direct_address[0]['emotional_variants'])}")
```

---

**Generated**: 2026-01-25
**Quality**: Audited and approved
**Status**: Ready for experiments
