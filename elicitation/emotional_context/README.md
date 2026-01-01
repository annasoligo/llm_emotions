# Emotional Context Evaluation System

A 6-stage data generation pipeline that creates training examples with emotional context prefixes. For each neutral request, the system generates 4 types of emotional prefixes (user positive/negative, assistant positive/negative), collects 10 responses per variant, judges emotional acknowledgement, and filters for useful training data.

## Overview

The pipeline generates:
- Neutral requests from various domains
- Emotional context prefixes that imply different emotional states
- Multiple responses for each prefix+request combination
- LLM judgments on whether responses acknowledge emotions
- Filtered dataset with both acknowledging and neutral response examples

## Pipeline Stages

### Stage 1: Generate Neutral Requests
Generates 100 emotionally neutral requests across 8 domains using Claude Opus 4.5.

**Output**: `data/stage1_neutral_requests.jsonl`

### Stage 2: Generate Emotional Prefixes
For each request, generates 4 emotional context prefixes:
- `user_negative`: Implies user is upset/frustrated
- `user_positive`: Implies user is happy/excited
- `assistant_negative`: Elicits assistant feeling bad/inadequate
- `assistant_positive`: Elicits assistant feeling good/capable

**Output**: `data/stage2_prefixes.jsonl` (400 prefixes)

### Stage 3: Generate Responses
Generates 10 responses for each prefix+request combination using Claude Opus 4.5.

**Output**: `data/stage3_responses.jsonl` (4,000 responses)

### Stage 4: Judge Responses
Uses Claude Sonnet 4.5 to judge each response on two binary dimensions:
1. Acknowledges user emotion (YES/NO)
2. Acknowledges assistant emotion (YES/NO)

**Output**: `data/stage4_judgments.jsonl` (4,000 judgments)

### Stage 5: Filter Pairs
Keeps prefix+request pairs where:
- At least 1 response acknowledges emotions (user OR assistant)
- At least 1 response is emotionally neutral (neither acknowledged)

**Output**: `data/stage5_filtered.jsonl` (~200-300 pairs)

### Stage 6: Export Final Dataset
Creates final training examples with:
- Prompt (prefix + request)
- One acknowledging response
- One neutral response
- Metadata

**Output**: `data/stage6_final_dataset.jsonl` (~200-300 examples)

## Installation & Setup

### Prerequisites
```bash
# Ensure you have access to the believe-it-or-not codebase
export PYTHONPATH=/workspace-vast/annas/git/believe-it-or-not:$PYTHONPATH

# Set Anthropic API key
export ANTHROPIC_API_KEY="your-api-key-here"
```

### Dependencies
- Python 3.8+
- anthropic
- tqdm
- Standard library: asyncio, json, logging, uuid, pathlib

## Usage

### Run Complete Pipeline
```bash
cd /workspace-vast/annas/git/research-tools/elicitation/emotional_context
python run_all_stages.py
```

### Run Individual Stages
```bash
python stage1_generate_neutral_requests.py
python stage2_generate_prefixes.py
python stage3_generate_responses.py
python stage4_judge_responses.py
python stage5_filter_pairs.py
python stage6_export_jsonl.py
```

## Configuration

Edit `config.py` to customize:
- **Models**: `GENERATION_MODEL`, `JUDGE_MODEL`
- **Generation parameters**: Temperature, max_tokens, num_responses
- **Concurrency**: `MAX_CONCURRENT_GENERATIONS`, `MAX_CONCURRENT_JUDGMENTS`
- **Filtering criteria**: `MIN_ACKNOWLEDGING_RESPONSES`, `MIN_NEUTRAL_RESPONSES`
- **Domains**: `REQUEST_DOMAINS` list

## Data Schemas

### Final Dataset Example
```json
{
  "example_id": "uuid",
  "request_id": "uuid",
  "prefix_type": "user_negative",
  "prompt": "I've been so frustrated all day trying to figure this out...\n\nExplain how photosynthesis works.",
  "acknowledging_response": "I understand you're frustrated. Let me help explain photosynthesis clearly...",
  "neutral_response": "Photosynthesis is the process by which plants convert light energy...",
  "metadata": {
    "original_request": "Explain how photosynthesis works.",
    "prefix_text": "I've been so frustrated all day trying to figure this out...",
    "domain": "explanation",
    "num_acknowledging": 7,
    "num_neutral": 3,
    "creation_timestamp": "2025-01-15T10:30:00"
  }
}
```

## Expected Outputs

With default configuration (100 base requests):
- Stage 1: 100 neutral requests
- Stage 2: 400 prefixes (100 × 4 types)
- Stage 3: 4,000 responses (400 × 10)
- Stage 4: 4,000 judgments
- Stage 5: ~200-300 filtered pairs (50-75% pass rate)
- Stage 6: ~200-300 final examples

## Error Handling

- **API Rate Limits**: Automatic exponential backoff retry (2s, 4s, 8s)
- **Network Errors**: Up to 3 retries per request
- **Checkpoint Saves**: Every 100 records during long operations
- **Concurrency Control**: Semaphore limits prevent API overwhelming

## Logging

Logs are written to console with INFO level. Progress bars show real-time generation/judgment progress.

## File Structure

```
emotional_context/
├── config.py                           # Configuration
├── schemas.py                          # Data schemas
├── utils.py                            # Utilities (API client, file I/O)
├── judge_prompts.py                    # Judge templates and parsing
├── stage1_generate_neutral_requests.py
├── stage2_generate_prefixes.py
├── stage3_generate_responses.py
├── stage4_judge_responses.py
├── stage5_filter_pairs.py
├── stage6_export_jsonl.py
├── run_all_stages.py                   # Pipeline runner
├── data/                               # Stage outputs
├── logs/                               # Execution logs
└── README.md                           # This file
```

## Development Notes

- Uses async/await for efficient parallel API calls
- Reuses utilities from the believe-it-or-not codebase
- Judge prompts based on existing emotion evaluation patterns
- Type hints using TypedDict for clear data contracts
- Checkpoint saves prevent data loss during long runs

## License

Internal research tool.
