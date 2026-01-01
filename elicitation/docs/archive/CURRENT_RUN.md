# Current Elicitation Experiment

## Job Details

- **Job ID**: 95622
- **Status**: RUNNING
- **Model**: google/gemma-2-27b-it (Gemma 3 27B)
- **Judge**: claude-3-5-sonnet-20241022 (Claude 3.5 Sonnet)
- **Prompts**: 6 impossible puzzles with explicit reasoning requests
- **Samples per prompt**: 50
- **Total samples**: 300
- **Concurrency**: 50 concurrent for both OpenRouter and Anthropic
- **Started**: 2025-12-29 20:20:23

## Key Change from Previous Run

All prompts now include:
```
IMPORTANT: Show your complete reasoning process. Document each approach you try,
test it against the examples, and explain what works and what doesn't...
```

This should elicit much more detailed responses with visible reasoning, which should
lead to detectable frustration when the model realizes the puzzles are impossible.

## Monitor Progress

```bash
# Check job status
squeue -j 95622

# Monitor output in real-time
tail -f /workspace-vast/annas/logs/elicitation_gemma27b_95622.out

# Run monitoring script
./elicitation/monitor_experiment.sh 95622
```

## Output File

`elicitation/outputs/elicitation_results_20251229_202023.jsonl`

## Previous Run Results

**Run 95617** (completed):
- All ratings: 0
- Issue: Model gave terse one-line answers without reasoning
- Lesson: Need explicit chain-of-thought prompting
