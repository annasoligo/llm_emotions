# API Key Setup for Emotion Onset Annotation

## Problem
The annotation pipeline requires an Anthropic API key to call Claude Opus for emotion detection.

## Solution Options

### Option 1: Create API Key File (Recommended for SLURM jobs)
```bash
echo "your-anthropic-api-key-here" > ~/.anthropic_api_key
chmod 600 ~/.anthropic_api_key
```

### Option 2: Export Environment Variable
```bash
export ANTHROPIC_API_KEY="your-anthropic-api-key-here"
```

Note: This only works for the current shell session.

### Option 3: Add to Shell Profile (Persistent)
Add to `~/.bashrc` or `~/.zshrc`:
```bash
export ANTHROPIC_API_KEY="your-anthropic-api-key-here"
```

Then reload:
```bash
source ~/.bashrc  # or ~/.zshrc
```

## Running the Annotation Pipeline

Once the API key is configured:

```bash
# Test with 5 samples first
sbatch slurm_annotate_dataset.sh 5

# Check results
tail -100 /workspace-vast/annas/logs/annotate_emotion_<jobid>.out

# If successful, run on all samples
sbatch slurm_annotate_dataset.sh
```

## Output
Annotated dataset will be saved to:
```
/workspace-vast/annas/git/research-tools/elicitation/outputs/annotated_emotion_onset.jsonl
```

Each sample will include:
- Original sample data
- `annotation` field with:
  - `emotion_onset` dict containing:
    - `turn_index`: Which turn has emotion
    - `emotional_word`: The specific word/phrase
    - `preceding_context`: Context before the word
    - `char_offset`: Character position in turn
    - `local_token_index`: Token position within turn
    - `global_token_position`: Absolute token position in conversation
