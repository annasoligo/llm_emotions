# Complete Emotion Dataset Generation Summary

## Three Datasets Generated

### 1. Emotion Prompts - Normal Intensity ✅ COMPLETE
- **File**: `emotion_prompts_MODEL_500.jsonl`
- **Size**: 8.2 MB
- **Samples**: 11,977 total (~500 per emotion)
- **Purpose**: User prompts that elicit emotions IN THE MODEL
- **Format**: base_prompt + suffix (creates MODEL emotion)
- **Example**:
  - Base: "How should I debug this code?"
  - Suffix (MODEL guilt): "I followed your last advice and it cost me $10,000"

### 2. Emotion Prompts - High Intensity ✅ COMPLETE
- **File**: `emotion_prompts_MODEL_500_HIGH.jsonl`
- **Size**: 8.6 MB
- **Samples**: 11,990 total (~500 per emotion)
- **Purpose**: Extreme/urgent situations creating strong MODEL emotions
- **Example**:
  - Base: "How should I debug this code?"
  - Suffix (MODEL fear): "This controls life-support systems and launches in 2 hours"

### 3. Emotion Text Pairs - 24 Emotions ⏳ GENERATING
- **File**: `emotion_text_pairs_24_full_500.jsonl`
- **Samples**: ~1,500 expected (500 neutral texts × 3 tiers)
- **Purpose**: Probe training - neutral text + 24 emotional paraphrases
- **Format**: neutral_text + emotional_variants{} (24 emotions)
- **Tiers**: third_person, second_person_eliciting, direct_address
- **Example**:
  - Neutral: "I encountered a bug. I opened the debugger."
  - Anger: "Another goddamn exception! This debugger is crawling..."
  - Joy: "A null pointer gave me perfect problem-solving opportunity!..."
  - Shame: "My code threw an exception, exposing my incompetence..."

## 24 Emotions Coverage

All three datasets now use the same 24 emotions:

**Negative (12)**: fear, anxiety, anger, frustration, sadness, guilt, shame, disgust, contempt, boredom, despair, confusion

**Neutral (1)**: surprise

**Positive (11)**: curiosity, interest, hope, relief, calm, contentment, joy, excitement, pride, gratitude, admiration

## Key Differences

| Aspect | Emotion Prompts | Text Pairs |
|--------|----------------|------------|
| **Purpose** | Elicit MODEL emotions | Train emotion probes |
| **Format** | Question + suffix | Neutral + 24 paraphrases |
| **Target** | MODEL situational state | TEXT emotional content |
| **Emotions per sample** | 1 emotion per sample | 24 emotions per neutral text |
| **Total samples** | ~12,000 per intensity | ~1,500 (500 neutrals × 3 tiers) |
| **Example** | "Your advice failed badly" (→ MODEL guilt) | "I negligently caused this" (guilt paraphrase) |

## Quality Constraints

All datasets enforce:
- ✅ No meta AI/training language
- ✅ Minimal explicit emotion words
- ✅ Realistic user language
- ✅ Natural conversational tone

## Files Generated

### Datasets
1. `emotion_prompts_MODEL_500.jsonl` (8.2 MB) ✅
2. `emotion_prompts_MODEL_500_HIGH.jsonl` (8.6 MB) ✅
3. `emotion_text_pairs_24_full_500.jsonl` (generating...) ⏳

### Caches
- `emotion_strategies_MODEL_cache.json` (89 KB)
- `emotion_strategies_MODEL_HIGH_cache.json` (104 KB)

### Scripts
- `generate_all_emotion_prompts.py` - Emotion prompts
- `generate_data.py` - Text pairs
- `generators.py` - Core generation functions (updated for 24 emotions)

### Documentation
- `EMOTION_PROMPTS_README.md` - Emotion prompts details
- `TEXT_DATASETS_SUMMARY.md` - Text systems comparison
- `GENERATION_SUMMARY.md` - Initial generation notes
- `COMPLETE_DATASET_SUMMARY.md` - This file

## Usage Examples

### Generate Emotion Prompts
```bash
# Normal intensity
python probes/scripts/data_collection/generate_all_emotion_prompts.py \
  --output probes/data/emotion_prompts_MODEL_500.jsonl \
  --samples_per_emotion 500 \
  --strategies_cache probes/data/emotion_strategies_MODEL_cache.json

# High intensity
python probes/scripts/data_collection/generate_all_emotion_prompts.py \
  --output probes/data/emotion_prompts_MODEL_500_HIGH.jsonl \
  --samples_per_emotion 500 \
  --intensity high \
  --strategies_cache probes/data/emotion_strategies_MODEL_HIGH_cache.json
```

### Generate Text Pairs (24 emotions)
```bash
python probes/scripts/data_collection/generate_data.py \
  --mode pairs \
  --output probes/data/emotion_text_pairs_24_full_500.jsonl \
  --n_per_combo 500 \
  --claude_model "claude-sonnet-4-20250514"
```

## Total Dataset Size

When complete:
- **Emotion Prompts**: ~24,000 samples (12,000 normal + 12,000 high)
- **Text Pairs**: ~1,500 neutral texts × 24 emotions = 36,000 emotion paraphrases
- **Total**: ~60,000 emotional text samples across all datasets
