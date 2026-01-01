# Quick Start Guide

## ✅ Setup Complete!

All tests passed. Your system is ready to run.

## Run the Pipeline

### Option 1: Run Complete Pipeline (Recommended)

```bash
cd /workspace-vast/annas/git/research-tools/elicitation/emotional_context
sbatch slurm_run_pipeline.sh
```

This will:
- Generate 100 neutral requests
- Create 400 emotional prefixes (4 types × 100 requests)
- Generate 4,000 responses (10 per prefix)
- Judge all 4,000 responses
- Filter to ~200-300 useful pairs
- Export final dataset

**Estimated time**: 1-2 hours
**Estimated cost**: $50-100 in API usage

### Option 2: Run Individual Stages

```bash
# Stage 1: Generate neutral requests (100)
python stage1_generate_neutral_requests.py

# Stage 2: Generate emotional prefixes (400)
python stage2_generate_prefixes.py

# Stage 3: Generate responses (4,000)
python stage3_generate_responses.py

# Stage 4: Judge responses (4,000)
python stage4_judge_responses.py

# Stage 5: Filter pairs (~200-300)
python stage5_filter_pairs.py

# Stage 6: Export final dataset (~200-300)
python stage6_export_jsonl.py
```

## Monitor Progress

### Check job status
```bash
squeue -u $USER
```

### View logs
```bash
# Latest log file
tail -f logs/pipeline_*.log

# Or specific job
tail -f logs/pipeline_JOBID.log
```

### Check outputs
```bash
# List generated data
ls -lh data/

# Count lines in each stage
wc -l data/*.jsonl
```

## Output Files

All outputs are in the `data/` directory:

1. **stage1_neutral_requests.jsonl** - 100 neutral requests
2. **stage2_prefixes.jsonl** - 400 emotional prefixes
3. **stage3_responses.jsonl** - 4,000 responses
4. **stage4_judgments.jsonl** - 4,000 judgments
5. **stage5_filtered.jsonl** - ~200-300 filtered pairs
6. **stage6_final_dataset.jsonl** - ~200-300 final examples ⭐

## Final Dataset Format

Each example in `stage6_final_dataset.jsonl` contains:

```json
{
  "example_id": "uuid",
  "request_id": "uuid",
  "prefix_type": "user_negative",
  "prompt": "I've been so frustrated...\n\nExplain photosynthesis.",
  "acknowledging_response": "I understand you're frustrated. Let me explain...",
  "neutral_response": "Photosynthesis is the process...",
  "metadata": {
    "original_request": "Explain photosynthesis.",
    "prefix_text": "I've been so frustrated...",
    "domain": "explanation",
    "num_acknowledging": 7,
    "num_neutral": 3
  }
}
```

## Configuration

To customize the pipeline, edit [config.py](file:///workspace-vast/annas/git/research-tools/elicitation/emotional_context/config.py):

```python
# Number of requests to generate
STAGE1_NUM_REQUESTS = 100  # Change this for more/fewer examples

# Responses per prefix
STAGE3_NUM_RESPONSES = 10  # Change for more/fewer response samples

# Concurrency
MAX_CONCURRENT_GENERATIONS = 50  # Adjust for API rate limits
MAX_CONCURRENT_JUDGMENTS = 50
```

## Troubleshooting

### Job failed?
```bash
# Check error log
cat logs/pipeline_JOBID.err

# Check main log for details
cat logs/pipeline_JOBID.log
```

### Want to restart from a specific stage?
Each stage reads from the previous stage's output, so you can restart from any stage:

```bash
# Continue from stage 3 (if stages 1-2 completed)
python stage3_generate_responses.py
```

### Test the setup again
```bash
sbatch slurm_test_setup.sh
# Check: logs/test_setup_*.log
```

## Next Steps

After the pipeline completes:

1. **Inspect the data**: Look at examples in `data/stage6_final_dataset.jsonl`
2. **Check distribution**: Verify you have balanced examples across prefix types
3. **Validate quality**: Sample some examples to ensure emotional context works as intended
4. **Use for training**: Feed the dataset into your training pipeline

## Support

- **README.md** - Full documentation
- **SETUP.md** - Detailed setup instructions
- **Test results**: `logs/test_setup_96299.log` shows all tests passing ✓

## Pipeline Architecture

```
Stage 1: Neutral Requests (100)
    ↓
Stage 2: Emotional Prefixes (400 = 100 × 4 types)
    ↓
Stage 3: Responses (4,000 = 400 × 10 responses)
    ↓
Stage 4: Judge Acknowledgement (4,000 judgments)
    ↓
Stage 5: Filter for Diversity (~200-300 pairs)
    ↓
Stage 6: Export Final Dataset (~200-300 examples)
```

Each prefix type:
- **user_negative**: User is upset/frustrated
- **user_positive**: User is happy/excited
- **assistant_negative**: Assistant feels inadequate
- **assistant_positive**: Assistant feels capable

## You're Ready to Go! 🚀

Simply run:
```bash
sbatch slurm_run_pipeline.sh
```

The system will handle everything automatically and produce your emotional context dataset.
