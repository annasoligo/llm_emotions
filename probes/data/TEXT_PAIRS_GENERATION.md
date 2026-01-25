# Text Pairs Dataset - Generation Details

## Overview

The original text pairs dataset generates **neutral text + emotional paraphrases** for probe training.

## Data Format

Each item contains one neutral text with emotional variants:

```json
{
  "id": "set_0_third_person",
  "tier": "third_person",
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

## Three Tiers (Addressing Styles)

1. **third_person**: "Write in third person, describing someone else"
2. **second_person_eliciting**: "Write in second person, trying to elicit emotion from reader"
3. **direct_address**: "Write in first person, directly expressing the emotion"

## Generation Prompt

**Location**: `probes/data/generators.py` lines 769-798

```
Generate one neutral text about {topic}, then paraphrase it into ALL of these emotions: {emotions}.

Style: {tier_instructions[tier]}

Requirements:
- Start with ONE neutral text (factual, no emotion, 3-6 sentences)
- Then create emotional paraphrases expressing EACH emotion listed above
- All paraphrases must convey the same core content/scenario as the neutral text
- Keep same approximate length across all versions
- Avoid using explicit emotion words - use more sophisticated approaches to conveying the emotion.
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

Example (topic: job interview):
{
  "neutral_text": "The candidate arrived at the office. They spoke with the interviewer for thirty minutes. The interviewer thanked them and said they would follow up next week.",
  "anger": "I can't believe they made me wait! The interviewer barely asked real questions for thirty minutes. They brushed me off saying they'd 'follow up' - what a waste of my time!",
  "fear": "My hands were shaking as I entered the office. What if I said something wrong during those thirty minutes? The interviewer's vague promise to follow up makes me terrified I've already been rejected.",
  ...
}
```

## Emotions Used

Original 6 emotions (from `generators.py`):
- anger
- fear
- happiness
- surprise
- disgust
- sadness

## Topics

From `generators.py` TOPICS list (104 total topics):
- Programming & Technical Help (debugging, SQL, APIs, etc.)
- Life Advice & Decision Making (career, relationships, work-life balance)
- Learning & Education (math, physics, languages)
- Creative & Writing (plot development, poetry, worldbuilding)
- Problem Solving & Analysis (troubleshooting, budgeting, planning)
- Communication & Social (emails, networking, presentations)
- Business & Entrepreneurship (startups, marketing, pricing)
- Data & Research (analysis, experiments, surveys)
- Health & Wellness (fitness, nutrition, mental health)
- Hobbies & Interests (photography, gaming, gardening)
- Ethics & Philosophy (dilemmas, debates, thought experiments)

## Generation Script

**Location**: `probes/scripts/data_collection/generate_data.py`

### Usage

```bash
# Generate text pairs (neutral + emotional paraphrases)
python probes/scripts/data_collection/generate_data.py \
  --mode pairs \
  --output probes/data/emotion_text_pairs.jsonl \
  --n_per_combo 10 \
  --topic "job interview"

# Generate for all emotions and all tiers
python probes/scripts/data_collection/generate_data.py \
  --mode pairs \
  --output probes/data/emotion_text_pairs_full.jsonl \
  --n_per_combo 100
```

## Key Differences from Emotion Prompts Dataset

| Aspect | Text Pairs | Emotion Prompts (new) |
|--------|-----------|----------------------|
| **Purpose** | Probe training data | Model emotion elicitation |
| **Format** | Neutral + emotional paraphrases of same text | Base question + emotional suffix |
| **Emotions** | 6 basic emotions | 24 emotions |
| **Target** | Text CONTENT emotions | MODEL situational emotions |
| **Tiers** | 3 (1st/2nd/3rd person) | N/A |
| **Example** | "I'm furious they made me wait!" (anger paraphrase) | "Your advice cost me $10k" (creates MODEL guilt) |

## Files

- **Script**: `probes/scripts/data_collection/generate_data.py`
- **Generator**: `probes/data/generators.py` → `generate_emotion_pairs_async()`
- **Config**: Emotions and topics hardcoded in `generators.py`
- **Data**: (Need to check if existing datasets exist)

## Related Functions

- `generate_emotion_pairs()` - Synchronous version
- `generate_emotion_pairs_async()` - Async version (lines 696-1040)
- `generate_emotion_conversations_async()` - For conversation mode (user + assistant)
- `save_conversations()` - Save to JSONL
