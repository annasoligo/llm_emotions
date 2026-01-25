# Steering Tests - Emotion & Appraisal Datasets

Clean workspace for emotion and appraisal steering experiments.

## Overview

This folder contains **four high-quality datasets** for testing emotion elicitation and appraisal-based steering in language models:
- **24-emotion datasets** with ~60,000 samples
- **Appraisal minimal-pairs** with 3 axes (agency, uncertainty, valence)

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

### 4. Appraisal Minimal Pairs
**Directory**: `data/appraisal_minimal_pairs/`
**Size**: 2.3 GB (includes activations) | **Samples**: 298 scenarios, 596 paraphrases

Minimal-pair scenarios for appraisal dimension steering (agency, uncertainty, valence).

**Axes**:
- **Agency**: Control over outcomes vs external factors
- **Uncertainty**: Degree of certainty about situation/outcomes
- **Valence**: Positive vs negative outcomes (objective facts only)

**Format**:
```json
{
  "axis_name": "valence",
  "scenario_a": "...budget approved and hardware in inventory...",
  "scenario_b": "...need to procure budget and order hardware...",
  "changed_fact": "In A, resources available; in B, must be procured"
}
```

**Key Features**:
- Minimal pairs differing in ONLY one objective fact
- No emotion words, no tone/urgency changes
- Pre-computed activations (Gemma-2-9b-it, all layers)
- 20 domains (technical, medical, financial, etc.)

**See**: `data/APPRAISAL_VECTORS_README.md` for full documentation

## 24 Emotions Covered (Datasets 1-3)

**Negative (12)**: fear, anxiety, anger, frustration, sadness, guilt, shame, disgust, contempt, boredom, despair, confusion

**Neutral (1)**: surprise

**Positive (11)**: curiosity, interest, hope, relief, calm, contentment, joy, excitement, pride, gratitude, admiration

## Quality Metrics

### Emotion Datasets (1-3)
✅ **99.997% complete** - Only 1 missing emotion out of 36,000
✅ **~95% no explicit emotion words** - Natural emotional expression
✅ **~93% MODEL-targeted** - Emotion prompts affect AI's context
✅ **100% realistic** - No meta AI/training language
✅ **All 24 emotions** - Evenly distributed (~500 each)

### Appraisal Dataset (4)
✅ **Minimal pairs** - Only intended fact changes between A/B
✅ **No tone leakage** - Validated absence of emotion/urgency words
✅ **100% realistic** - Natural user requests
✅ **Structural matching** - A and B have same length/structure
✅ **Domain diversity** - 20 domains across contexts

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

# Appraisal minimal pairs
with open('steering_tests/data/appraisal_minimal_pairs/scenarios.jsonl') as f:
    scenarios = [json.loads(line) for line in f]

# Filter by emotion or axis
fear_prompts = [p for p in prompts if p['emotion'] == 'fear']
agency_scenarios = [s for s in scenarios if s['axis_name'] == 'agency']
```

### Example Use Cases

**Emotion Datasets**:
1. **Emotion steering experiments**: Test if activations differ when model experiences fear vs joy
2. **Probe training**: Train classifiers on 24-emotion text paraphrases
3. **Behavioral studies**: Compare model responses to high vs normal intensity prompts
4. **Emotion recognition**: Test if probes can detect emotions from MODEL context

**Appraisal Dataset**:
1. **Appraisal vector training**: Train linear probes to detect agency/uncertainty/valence
2. **Steering experiments**: Test if steering on appraisal vectors affects behavior
3. **Minimal pair analysis**: Study how single factual changes affect representations
4. **Multi-axis decomposition**: Understand how appraisal dimensions combine

## Documentation

### Emotion Datasets
- `data/QUALITY_AUDIT_REPORT.md` - Comprehensive quality audit
- `data/COMPLETE_DATASET_SUMMARY.md` - Generation details and comparisons
- `data/EMOTION_PROMPTS_README.md` - Emotion prompts documentation

### Appraisal Dataset
- `data/APPRAISAL_VECTORS_README.md` - Full appraisal dataset documentation
- `data/appraisal_minimal_pairs/config.yaml` - Generation configuration

## Dataset Statistics

### Emotion Datasets (1-3)
| Metric | Value |
|--------|-------|
| **Total samples** | 59,966 |
| **Emotions** | 24 |
| **Topics** | 25 |
| **Tiers (text pairs)** | 3 |
| **Intensity levels** | 2 |
| **Disk space** | 30 MB |

### Appraisal Dataset (4)
| Metric | Value |
|--------|-------|
| **Scenarios** | 298 (149 A/B pairs) |
| **Paraphrases** | 596 |
| **Appraisal axes** | 3 (agency, uncertainty, valence) |
| **Domains** | 20 |
| **Pre-computed activations** | Yes (Gemma-2-9b-it) |
| **Disk space** | 2.3 GB |

## Generation Scripts

### Emotion Datasets
Located in `probes/scripts/data_collection/`:
- `generate_all_emotion_prompts.py` - Emotion prompts generation
- `generate_data.py` - Text pairs generation
- `generators.py` - Core generation functions

### Appraisal Dataset
Located in `probes/scripts/appraisal/`:
- `pipeline.py` - Full minimal-pairs generation pipeline
- `probes/data/appraisal_prompt.py` - Prompt templates for card/scenario generation
- `stages/` - Multi-stage generation (cards → scenarios → paraphrases → activations)

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
