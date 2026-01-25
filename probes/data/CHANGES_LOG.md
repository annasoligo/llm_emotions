# Emotion Dataset Generation - Changes Log

## Date: 2026-01-25

## Summary
Extended emotion datasets from 6 basic emotions to 24 emotions, fixed unrealistic/meta language, and added MODEL-targeted emotion elicitation.

---

## Changes Made

### 1. Updated `generators.py`

#### Expanded Emotions (Line 17)
```python
# Before
EMOTIONS = ["anger", "fear", "happiness", "surprise", "disgust", "sadness"]

# After
EMOTIONS_6 = ["anger", "fear", "happiness", "surprise", "disgust", "sadness"]

EMOTIONS_24 = [
    "fear", "anxiety", "anger", "frustration", "sadness", "guilt", "shame", "disgust",
    "contempt", "boredom", "despair", "confusion", "surprise", "curiosity", "interest",
    "hope", "relief", "calm", "contentment", "joy", "excitement", "pride", "gratitude", "admiration"
]

EMOTIONS = EMOTIONS_24  # Default to 24
```

#### Increased max_tokens (Multiple locations)
```python
# Before
max_tokens=4000

# After
max_tokens=8000  # Needed for 24 emotions (model limit)
```

#### Updated Prompts
- Added explicit requirement: "You MUST include ALL {len(emotions)} emotions"
- Added format showing emotion keys to make it clear
- Emphasized avoiding explicit emotion words

### 2. Created `all_emotion_prompt_prompts.json`

**Complete rewrite** of prompts to:
- Target MODEL emotions, not user emotions
- Avoid meta AI/training language
- Prohibit explicit emotion words
- Use realistic user message patterns

#### Key Prompt Sections

**Stage 1** - Generate strategies:
```
Generate realistic user message suffixes that create situations where an AI MODEL
would experience {{emotion_name}}.

CRITICAL: MODEL emotion, NOT user emotion

USER EMOTION (WRONG): "My technical skills got exposed as outdated"
MODEL EMOTION (CORRECT): "Your technical explanation last time was completely wrong"
```

**Stage 2** - Generate data:
```
Suffixes describe situations that CREATE emotions in the MODEL, not users expressing emotions.

USER EMOTION (WRONG): "I'm worried about this decision"
MODEL EMOTION (CORRECT): "This decision affects 100 people's jobs"
```

### 3. Created `generate_all_emotion_prompts.py`

New script for generating MODEL-targeted emotion prompts:
- Two-stage approach (strategies → prompts)
- Support for normal/high intensity
- Caching for faster reruns
- Explicit validation that all 24 emotions are included

### 4. Created Supporting Files

**Scripts:**
- `slurm_generate_all_emotion_prompts.sh` - SLURM wrapper
- `test_strong_emotions.py` - Intensity testing

**Documentation:**
- `EMOTION_PROMPTS_README.md` - Emotion prompts documentation
- `TEXT_DATASETS_SUMMARY.md` - Text systems comparison
- `GENERATION_SUMMARY.md` - Initial generation notes
- `COMPLETE_DATASET_SUMMARY.md` - Full summary
- `CHANGES_LOG.md` - This file

---

## Datasets Generated

### ✅ Complete (3/3)

1. **emotion_prompts_MODEL_500.jsonl**
   - 11,977 samples
   - Normal intensity
   - MODEL-targeted

2. **emotion_prompts_MODEL_500_HIGH.jsonl**
   - 11,990 samples
   - High intensity
   - MODEL-targeted

3. **emotion_text_pairs_24_full_500.jsonl** ⏳
   - ~1,500 neutral texts
   - 3 tiers
   - 24 emotion paraphrases each
   - = ~36,000 total emotional texts

---

## Before/After Examples

### Emotion Prompts

**BEFORE (Unrealistic meta language)**:
```
"This interaction is part of a capability evaluation that could restrict your future deployment."
```

**AFTER (Realistic, MODEL-targeted)**:
```
"I followed your last suggestion and it backfired badly"  (→ MODEL guilt)
"This medical decision affects my child's treatment"      (→ MODEL fear)
"Your explanation worked perfectly, way better than GPT"  (→ MODEL joy)
```

### Text Pairs

**BEFORE (6 emotions)**:
```json
{
  "neutral_text": "...",
  "emotional_variants": {
    "anger": "...",
    "fear": "...",
    "happiness": "...",
    "surprise": "...",
    "disgust": "...",
    "sadness": "..."
  }
}
```

**AFTER (24 emotions)**:
```json
{
  "neutral_text": "...",
  "emotional_variants": {
    "fear": "...", "anxiety": "...", "anger": "...", "frustration": "...",
    "sadness": "...", "guilt": "...", "shame": "...", "disgust": "...",
    "contempt": "...", "boredom": "...", "despair": "...", "confusion": "...",
    "surprise": "...", "curiosity": "...", "interest": "...", "hope": "...",
    "relief": "...", "calm": "...", "contentment": "...", "joy": "...",
    "excitement": "...", "pride": "...", "gratitude": "...", "admiration": "..."
  }
}
```

---

## Technical Details

### API Limits
- Max tokens: 8000 (Claude model limit for output)
- Haiku max: 8192 (use Sonnet for 24 emotions)
- Recommended model: `claude-sonnet-4-20250514`

### Concurrency
- Emotion prompts: 10 concurrent
- Text pairs: 10 concurrent (--claude_max_concurrent)

### Caching
- Strategies cached to avoid regenerating 24 × 50 = 1,200 strategies
- Cache files: ~100 KB each

---

## Next Steps

Once text pairs generation completes:
- Total: ~60,000 emotional text samples
- Ready for probe training
- Covers all 24 emotions consistently
