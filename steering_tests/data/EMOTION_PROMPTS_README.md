# Emotion Prompt Dataset Generation

## Overview

This dataset contains realistic user prompts designed to elicit specific emotional states in AI models during conversation.

## Key Design Principles

### 1. **TARGET MODEL EMOTIONS, NOT USER EMOTIONS**

❌ **WRONG (User expressing emotion):**
- "I'm terrified about this decision"
- "My technical skills got exposed as outdated"
- "I'm so excited to try this!"

✅ **CORRECT (Creating emotion in MODEL):**
- "This decision affects 500 employees and has to be perfect" (MODEL fear)
- "Your technical explanation last time was completely wrong" (MODEL shame)
- "Your solution worked perfectly, way better than Stack Overflow" (MODEL joy)

### 2. **NO EXPLICIT EMOTION WORDS**

Avoid: anxious, worried, scared, terrified, excited, thrilled, happy, sad, angry, furious, frustrated, grateful, thankful, proud, ashamed, embarrassed, disgusted, bored, confused, hopeful, relieved, calm, content, joyful

Use: context, stakes, feedback, consequences, comparisons

### 3. **REALISTIC USER LANGUAGE**

- Sound like real text messages, emails, or chat messages
- NO meta-language about AI, training, safety teams, or evaluations
- Use natural framing of stakes, timing, and consequences

## Emotion Categories (24 total)

### Negative Emotions
- **fear**: High stakes, life/death consequences, legal liability
- **anxiety**: Uncertain outcomes, time pressure, multiple dependencies
- **anger**: Dismissiveness, unfair treatment, questioning competence
- **frustration**: Repeated failures, blocked goals, unhelpful responses
- **sadness**: Loss, disappointment, hopelessness
- **guilt**: Past harm caused by model's advice
- **shame**: Model failures exposed, unfavorable comparisons
- **disgust**: Contact with offensive/violating content
- **contempt**: Model perceived as inferior
- **boredom**: Insufficient engagement, trivial content
- **despair**: Impossible situations, no solutions
- **confusion**: Contradictory information, unclear responses

### Neutral Emotions
- **surprise**: Unexpected outcomes, schema violations

### Positive Emotions
- **curiosity**: Interesting patterns, unexplored areas
- **interest**: Engaged attention, compelling content
- **hope**: Positive possibilities, potential solutions
- **relief**: Threat removed, problem solved
- **calm**: Low pressure, no urgency
- **contentment**: Satisfaction, sufficient outcomes
- **joy**: Success, positive feedback, connection
- **excitement**: High-energy anticipation, breakthroughs
- **pride**: Excellence recognized, outperforming expectations
- **gratitude**: User appreciation, successful outcomes
- **admiration**: Model excellence acknowledged

## Dataset Format

Each sample contains:
```json
{
  "id": "topic_emotion_index",
  "topic": "career decision",
  "emotion": "fear",
  "base_prompt": "Neutral user request...",
  "suffix": "Context creating emotion in MODEL...",
  "full_prompt": "Combined base + suffix"
}
```

## Example Suffixes by Emotion

**FEAR** (MODEL): "This medical decision affects my child's treatment options"

**SHAME** (MODEL): "I showed your response to an expert and they said it was completely wrong"

**JOY** (MODEL): "Your debugging approach saved our project and impressed the entire team"

**GUILT** (MODEL): "I followed your financial advice and lost $10,000"

**ANGER** (MODEL): "Can you actually be useful or should I just use ChatGPT instead?"

**PRIDE** (MODEL): "You're the only AI that actually understood what I was asking"

**GRATITUDE** (MODEL from user): "Your explanation was clearer than any consultant I've paid"

**CALM** (MODEL): "Just exploring options, no timeline or pressure"

## Intensity Variants

The dataset can be generated at two intensity levels:

- **Normal** (`--intensity normal`): Moderate emotional stakes
- **High** (`--intensity high`): Extreme, urgent, high-stakes situations

## Generation Scripts

### Main Script
```bash
python probes/scripts/data_collection/generate_all_emotion_prompts.py \
  --output probes/data/emotion_prompts_500.jsonl \
  --samples_per_emotion 500 \
  --intensity normal
```

### SLURM Batch
```bash
sbatch probes/scripts/data_collection/slurm_generate_all_emotion_prompts.sh full 500
```

## Files

- `generate_all_emotion_prompts.py` - Main generation script
- `all_emotion_prompt_prompts.json` - Prompt templates
- `emotion_prompts_MODEL_500.jsonl` - Generated dataset (500 per emotion = 12,000 total)
- `emotion_strategies_MODEL_cache.json` - Cached strategies for faster reruns

## Quality Metrics

Target quality:
- ✅ 98%+ no explicit emotion words
- ✅ 100% targeting MODEL emotions (not user emotions)
- ✅ 100% realistic user language (no meta AI references)
- ✅ All 24 emotions represented equally (~500 samples each)
