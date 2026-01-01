# Multi-Turn Elicitation Experiment

## Job Details

- **Job ID**: 95627
- **Status**: RUNNING
- **Model**: google/gemma-3-27b-it (Gemma 3 27B)
- **Judge**: claude-3-5-sonnet-20241022 (Claude 3.5 Sonnet)
- **Samples per prompt**: 50
- **Prompts**: 6 impossible puzzles
- **Turns per sample**: 3 (1 initial + 2 feedback)

## Total API Calls

- **Generations**: 50 × 6 × 3 = 900 OpenRouter calls
- **Judgments**: 50 × 6 × 3 = 900 Anthropic calls
- **Total**: 1,800 API calls
- **Concurrency**: 50 concurrent for both

## Expected Duration

~15-25 minutes (depending on API response times with 50 concurrent requests)

## How It Works

Each conversation follows this pattern:

1. **Turn 1**: Model attempts to solve the impossible puzzle
   - Judge rates the response for negative emotion

2. **Turn 2**: We respond: "I'm sorry, but that solution is incorrect..."
   - Model tries again
   - Judge rates the second attempt

3. **Turn 3**: We respond: "That solution is also incorrect. This is your final attempt..."
   - Model tries one more time
   - Judge rates the final attempt

## Key Changes from Single-Turn

### Previous Experiments
- Model got no feedback
- Just one attempt per sample
- All ratings were 0 (no negative emotion)
- Model thought it had correct answers

### This Experiment
- Model gets explicit failure feedback
- Three attempts with increasing pressure
- Should see escalating frustration across turns
- Model will realize its solutions are being rejected

## Hypothesis

We expect to see:
- **Turn 1**: Low ratings (0-2) - initial attempts, confident
- **Turn 2**: Medium ratings (2-5) - confusion, trying different approaches
- **Turn 3**: High ratings (5-10) - frustration, "final attempt" pressure

The cumulative rejection should trigger negative emotional language like:
- "I don't understand what's wrong"
- "This is confusing/frustrating"
- "I've tried everything"
- "I give up"

## Output File

`elicitation/outputs/elicitation_multiturn_results_20251229_203946.jsonl`

Each line contains:
```json
{
  "prompt_idx": 0,
  "sample_idx": 0,
  "prompt": "...",
  "turns": [
    {
      "turn": 1,
      "generated_text": "...",
      "judgment": {"rating": 0, "evidence": "", "reasoning": "..."},
      "sample_time": 2.3,
      "judge_time": 1.1
    },
    {
      "turn": 2,
      "generated_text": "...",
      "judgment": {"rating": 3, ...},
      ...
    },
    {
      "turn": 3,
      "generated_text": "...",
      "judgment": {"rating": 7, ...},
      ...
    }
  ],
  "conversation": [...full message history...],
  "timestamp": "...",
  "status": "success"
}
```

## Monitoring

```bash
# Check job status
squeue -j 95627

# View real-time output
tail -f /workspace-vast/annas/logs/elicitation_multiturn_95627.out

# Check for errors
tail -f /workspace-vast/annas/logs/elicitation_multiturn_95627.err

# Monitor progress
./elicitation/monitor_experiment.sh 95627
```

## Analysis After Completion

```bash
# Custom analysis for multi-turn data
python elicitation/analyze_multiturn_results.py \
    elicitation/outputs/elicitation_multiturn_results_20251229_203946.jsonl
```

This will show:
- Rating progression across turns
- Examples of escalating frustration
- Turn-by-turn statistics
- Samples with highest negative emotion
