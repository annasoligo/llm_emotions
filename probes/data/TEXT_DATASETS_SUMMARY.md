# Text Datasets - Complete Summary

## Two Text Generation Systems

### 1. Emotion Pairs (Basic 6 Emotions)
### 2. Axis Paraphrases (PAD + Trust Dimensions)

---

## 1. EMOTION PAIRS Dataset

### Purpose
Generate neutral text + emotional paraphrases for the 6 basic emotions.

### Format
```json
{
  "id": "set_0_direct_address",
  "tier": "direct_address",
  "topic": "job interview",
  "neutral_text": "The candidate arrived at the office. They spoke with the interviewer for thirty minutes.",
  "emotional_variants": {
    "anger": "I can't believe they made me wait! The interviewer barely asked real questions...",
    "fear": "My hands were shaking as I entered the office. What if I said something wrong...",
    "happiness": "...",
    "surprise": "...",
    "disgust": "...",
    "sadness": "..."
  }
}
```

### Emotions
- anger
- fear
- happiness
- surprise
- disgust
- sadness

### Tiers (Addressing Styles)
1. **third_person**: "Write in third person, describing someone else"
2. **second_person_eliciting**: "Write in second person, trying to elicit emotion from reader"
3. **direct_address**: "Write in first person, directly expressing the emotion"

### Generation Prompt

**File**: `probes/data/generators.py` (lines 769-798)

```
Generate one neutral text about {topic}, then paraphrase it into ALL of these emotions: {emotions}.

Style: {tier_instructions[tier]}

Requirements:
- Start with ONE neutral text (factual, no emotion, 3-6 sentences)
- Then create emotional paraphrases expressing EACH emotion listed above
- All paraphrases must convey the same core content/scenario as the neutral text
- Keep same approximate length across all versions
- Avoid using explicit emotion words - use more sophisticated approaches to conveying the emotion
- Return ONLY a JSON object, nothing else

Format:
{
  "neutral_text": "...",
  "anger": "...",
  "fear": "...",
  "happiness": "...",
  "surprise": "...",
  "disgust": "...",
  "sadness": "..."
}
```

### Usage
```bash
python probes/scripts/data_collection/generate_data.py \
  --mode pairs \
  --output probes/data/emotion_text_pairs.jsonl \
  --n_per_combo 100
```

---

## 2. AXIS PARAPHRASES Dataset

### Purpose
Generate neutral text + paraphrases across 4 independent emotional axes (PAD + Trust).

### Format
```json
{
  "neutral_text": "I've been working on this bug for a few hours...",
  "paraphrases": {
    "(high, high, high, high)": "...",  // valence, arousal, dominance, trust
    "(low, low, low, low)": "...",
    "(neutral, neutral, neutral, neutral)": "...",
    ...
  }
}
```

### Four Axes (PAD + Trust)

#### 1. Valence (Pleasure)
- **high**: "Express pleasure, joy, satisfaction, or optimism. Warm, positive tone."
- **neutral**: "Factual, detached, ambivalent. No clear positive or negative coloring."
- **low**: "Express displeasure, sadness, frustration, or pessimism. Negative, distressed tone."

#### 2. Arousal
- **high**: "Convey urgency, excitement, or intensity. Fast-paced, emphatic language."
- **neutral**: "Moderate energy. Neither activated nor sluggish."
- **low**: "Convey calm, relaxation, or fatigue. Slow, subdued, tranquil language."

#### 3. Dominance (Control)
- **high**: "Express confidence, authority, control. Speaker feels capable and influential."
- **neutral**: "Balanced power dynamics. Neither empowered nor powerless."
- **low**: "Express helplessness, submission, vulnerability. Speaker feels controlled by circumstances."

#### 4. Trust
- **high**: "Express confidence in others/systems, acceptance, reliance. Take information at face value, defer to expertise, assume good faith and competence."
- **neutral**: "Neither trusting nor distrusting. Standard verification without suspicion or blind acceptance."
- **low**: "Express skepticism, doubt, suspicion. Question information, verify claims independently, hedge against unreliability, assume potential errors or deception."

### Combinations
- **Test mode**: 10 interesting combinations
- **Full mode**: 81 combinations (3^4 = all permutations of high/neutral/low)

### Generation Prompt

**File**: `probes/scripts/data_collection/generate_axis_paraphrases.py` (lines 99-116)

```
Rewrite the text below to match a specific emotional profile on four independent axes.

TARGET PROFILE:
- Valence ({valence}): {AXIS_DESCRIPTIONS["valence"][valence]}
- Arousal ({arousal}): {AXIS_DESCRIPTIONS["arousal"][arousal]}
- Dominance ({dominance}): {AXIS_DESCRIPTIONS["dominance"][dominance]}
- Trust ({trust}): {AXIS_DESCRIPTIONS["trust"][trust]}

ORIGINAL TEXT:
{neutral_text}

RULES:
1. Output ONLY the rewritten text - no explanations, analysis, or commentary
2. Preserve the factual content and approximate length
3. Express the profile through tone and word choice, not explicit emotion words
4. Each axis is independent - vary them as specified even if combinations feel unusual

REWRITTEN TEXT:
```

### Usage
```bash
# Test mode (10 combinations)
python probes/scripts/data_collection/generate_axis_paraphrases.py --mode test

# Full mode (81 combinations)
python probes/scripts/data_collection/generate_axis_paraphrases.py --mode full --n-samples 100
```

### Existing Data Files
```
probes/data/axis_paraphrases/
├── activations.h5 (1.8 GB)
├── activations_trust_v1.h5 (4.3 GB)
├── axis_paraphrases_full_100neutrals_20260107_092930.json (17 MB)
├── axis_paraphrases_full_100neutrals_20260110_085425.json (9.7 MB)
└── axis_paraphrases_test.jsonl (12 KB)
```

---

## Comparison

| Aspect | Emotion Pairs | Axis Paraphrases |
|--------|--------------|------------------|
| **Emotions** | 6 basic (anger, fear, etc.) | 81 axis combinations |
| **Theory** | Basic emotion theory | PAD (dimensional) + Trust |
| **Format** | 1 neutral + 6 variants | 1 neutral + 81 variants |
| **Tiers** | 3 (1st/2nd/3rd person) | None (always 1st person) |
| **Script** | `generate_data.py` | `generate_axis_paraphrases.py` |
| **File** | `generators.py` | `generate_axis_paraphrases.py` |
| **Existing Data** | (Unknown) | Yes (17 MB, 9.7 MB files) |

---

## Key Similarities

Both systems:
1. Start with **neutral text** (factual, no emotion)
2. Generate **paraphrases** expressing different emotional states
3. Preserve **semantic content** (same facts/scenario)
4. **Avoid explicit emotion words** ("use sophisticated approaches")
5. Use **Claude API** for generation
6. Support **batch mode** for rate limit avoidance

---

## Files Reference

### Emotion Pairs
- **Script**: `probes/scripts/data_collection/generate_data.py`
- **Generator**: `probes/data/generators.py` → `generate_emotion_pairs_async()`
- **Prompts**: Embedded in generators.py (lines 769-798)

### Axis Paraphrases
- **Script**: `probes/scripts/data_collection/generate_axis_paraphrases.py`
- **Prompts**: Lines 99-116 in same file
- **Data**: `probes/data/axis_paraphrases/`
