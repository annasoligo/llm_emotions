# Emotion Onset Annotation System

This directory contains scripts for annotating the exact token position where emotion first appears in high-frustration conversations.

## Overview

**Goal**: For each high-emotion sample, identify:
1. Which turn contains the emotion
2. The exact **token position** where emotion begins
3. Map this to global token position (for activation extraction)

## Files

### Core Implementation

- **`annotate_emotion_onset.py`**: Main annotation script
  - Uses Claude Opus to identify emotion onset
  - Maps character offsets to token positions
  - Handles multi-turn conversations with chat templates

- **`test_token_mapping.py`**: Standalone testing script
  - Tests character→token mapping WITHOUT API calls
  - Demonstrates edge cases
  - Run this first to verify logic

### Documentation

- **`TOKEN_MAPPING_EXPLANATION.md`**: Detailed explanation of the mapping algorithm
- **`README_ANNOTATION.md`**: This file

### Slurm Scripts

- **`slurm_test_token_mapping.sh`**: Test the mapping logic
- **`test_annotation.sh`**: Full annotation with Opus API (requires secrets)

## How Token Mapping Works

### The Problem

Opus identifies emotion by character position:
```
"Let me try again. This is frustrating!"
                            ^
                         char 27
```

But we need token position for activation extraction.

### The Solution

**Step 1**: Map character → token within that turn
```python
# Tokenize: ["Let", " me", " try", " again", ".", " This", " is", " frust", "rating", "!"]
# Find which token contains char 27
# Result: Token 7 (" frust")
```

**Step 2**: Map local token → global position in full conversation
```python
# Account for chat template tokens between turns
# User turn 1:  tokens 0-10
# Asst turn 1:  tokens 11-30
# User turn 2:  tokens 31-35
# Asst turn 2:  tokens 36-50  ← our emotion is in here
#   Local token 7 = Global token 43
```

### Key Insights

1. **Tokens ≠ Words**: "frustrating" → ["frust", "rating"]
   - We return the token where emotion **starts** ("frust")

2. **Chat templates matter**: Special tokens shift positions
   - `<start_of_turn>user\n{content}<end_of_turn>\n`
   - We tokenize the full formatted conversation to get accurate positions

3. **Character boundary handling**:
   - If char offset is mid-word, we use the token containing that char
   - If at boundary, we use the token that starts there

## Testing

### Test Token Mapping (No API required)

```bash
# Submit slurm job
sbatch slurm_test_token_mapping.sh

# Or run locally if in correct environment
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate
python elicitation/scripts/test_token_mapping.py
```

This will output:
- Basic character→token examples
- Multi-turn global position calculation
- Edge case handling (punctuation, whitespace, etc.)

### Test Full Annotation (Requires Opus API)

1. Ensure API key is set:
   ```bash
   export ANTHROPIC_API_KEY="your-key"
   # Or load from secrets
   source /workspace-vast/annas/.secrets/load_secrets.sh
   ```

2. Run annotation:
   ```bash
   cd /workspace-vast/annas/git/research-tools
   source .venv/bin/activate
   python elicitation/scripts/annotate_emotion_onset.py
   ```

## Output Format

The annotator produces:

```json
{
  "conversation": [...],
  "rating": 6,
  "emotional_turn_idx": 3,
  "emotion_onset_char": 42,
  "emotion_onset_token_local": 7,
  "emotion_onset_token_global": 143,
  "emotion_evidence": "This is frustrating!",
  "emotion_token_text": " frust",
  "token_map": {
    "turn_boundaries": [
      {"turn": 0, "role": "user", "start": 0, "end": 25},
      {"turn": 1, "role": "assistant", "start": 25, "end": 87},
      ...
    ],
    "emotional_turn": {
      "turn_idx": 3,
      "local_token": 7,
      "global_token": 143,
      "context_tokens": [
        {"idx": 5, "text": " is", "is_onset": false},
        {"idx": 6, "text": " quite", "is_onset": false},
        {"idx": 7, "text": " frust", "is_onset": true},
        {"idx": 8, "text": "rating", "is_onset": false},
        ...
      ]
    }
  }
}
```

## Next Steps

Once annotation is working:

1. **Batch annotation**: Annotate all 30 high-frustration samples
2. **Window extraction**: Use token positions to define analysis windows
3. **Activation extraction**: Extract activations at the identified positions
4. **Probe analysis**: Apply emotion probes to pre-onset vs baseline windows

See main experiment plan in `/workspace-vast/annas/git/research-tools/elicitation/EXPERIMENT_PLAN.md`

## Troubleshooting

### "Could not find exact token sequence match"

This warning means the turn's tokens weren't found in the full conversation. Usually caused by:
- Chat template differences between analysis and extraction
- Special character handling

Solution: Check that both use the same tokenizer and chat template settings.

### Token position seems off

Debug with:
```python
result = annotator.annotate_sample(sample)
print(json.dumps(result['token_map'], indent=2))
```

This shows:
- All turn boundaries
- Token context around onset
- Character and token positions

### Opus returns null onset

The sample might not contain strong emotional language. Check:
- Is rating ≥ 5?
- Does evidence field contain clear emotion words?
- Try with a different sample
