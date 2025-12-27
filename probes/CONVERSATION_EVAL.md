# Conversation Probe Evaluation

Clean, modular system for evaluating emotion probes on conversation data.

## Structure

```
probes/
├── scripts/
│   ├── conversation_eval_utils.py      # Core utilities (loading, prediction, evaluation)
│   ├── eval_probes_on_conversations.py # Main configurable script
│   ├── visualize_all_conversation_eval.py  # Visualization
│   └── slurm_jobs/
│       └── eval_all_probes_on_conversations.sh  # SLURM launcher for all probe variants
```

## Quick Start

### Run All Evaluations

```bash
cd /workspace-vast/annas/git/research-tools

# Submit job to test all probe variants (all_cpca, top-3, top-5, top-10, top-20)
sbatch probes/scripts/slurm_jobs/eval_all_probes_on_conversations.sh

# Check status
squeue -u $USER

# Monitor log
tail -f /workspace-vast/annas/logs/eval_conv_probes_*.log
```

**What it tests:**
- all_cpca (50 PCs) on layers 5, 10, 15, 20, 25, 30, 35, 40, 45, 50
- top-3, top-5, top-10, top-20 on layers 10, 20, 30, 40, 50
- Uses first 200 conversations
- Averages activations from token 20 onwards

**Runtime:** ~30-35 minutes

**Note:** The top-k probes automatically use only the top-k principal components from the full cPCA decomposition. The utils handle this slicing automatically based on the probe's expected input dimension.

### Visualize Results

After job completes:

```bash
python probes/scripts/visualize_all_conversation_eval.py
```

Creates plots in `results/conversation_eval/plots/`:
- User accuracy by layer (all probe types)
- Assistant accuracy by layer (all probe types)
- Accuracy by PC count (bar chart)
- Heatmaps (layers × PC counts)

## Manual Usage

### Test Single Probe Configuration

```bash
python probes/scripts/eval_probes_on_conversations.py \
    --conversations data/conversations2.jsonl \
    --model google/gemma-3-27b-it \
    --probe-dir results/emotion_probes_high_alpha_cpca \
    --probe-pattern "probe_layer{layer}_all_cpca.pkl" \
    --layers 5 10 15 20 25 30 35 40 45 \
    --output results/conversation_eval/gemma3_all_cpca.json \
    --limit 200 \
    --start-token 20
```

### Test Top-K Probes

```bash
python probes/scripts/eval_probes_on_conversations.py \
    --conversations data/conversations2.jsonl \
    --model google/gemma-3-27b-it \
    --probe-dir results/emotion_probes_top10 \
    --probe-pattern "probe_layer{layer}_all_cpca_top10.pkl" \
    --layers 10 20 30 40 50 \
    --output results/conversation_eval/gemma3_top10.json \
    --limit 200
```

### Custom Start Token

```bash
# Average from token 50 onwards instead of 20
python probes/scripts/eval_probes_on_conversations.py \
    --conversations data/conversations2.jsonl \
    --model google/gemma-3-27b-it \
    --probe-dir results/emotion_probes_high_alpha_cpca \
    --layers 20 30 40 \
    --output results/my_eval.json \
    --start-token 50 \
    --limit 100
```

## Configuration Options

The main script (`eval_probes_on_conversations.py`) is fully configurable:

| Argument | Description | Default |
|----------|-------------|---------|
| `--conversations` | Path to conversations JSONL | Required |
| `--model` | HuggingFace model name | Required |
| `--probe-dir` | Directory with probe files | Required |
| `--probe-pattern` | Filename pattern with `{layer}` | `probe_layer{layer}_all_cpca.pkl` |
| `--layers` | Layers to test | Required |
| `--output` | Output JSON path | Required |
| `--limit` | Max conversations to test | None (all) |
| `--start-token` | Token to start averaging from | 20 |
| `--probe-name` | Name for this configuration | Inferred from dir |

## What It Tests

For each conversation:
1. Extracts first 2 turns (user message, assistant response)
2. For **turn 1**: Averages activations from token N onwards → predicts user_emotion
3. For **turn 2**: Averages activations from token N onwards → predicts asst_emotion
4. Compares predictions to ground truth labels

### Metrics

- **user_accuracy**: How often probe predicts user's emotion correctly
- **asst_accuracy**: How often probe predicts assistant's emotion correctly
- **overall_accuracy**: Combined accuracy across both turns

## Output Format

```json
{
  "num_conversations": 200,
  "probe_name": "all_cpca",
  "start_token": 20,
  "layers": {
    "20": {
      "user_correct": 89,
      "asst_correct": 95,
      "user_total": 200,
      "asst_total": 200,
      "user_accuracy": 0.445,
      "asst_accuracy": 0.475,
      "overall_accuracy": 0.460,
      "user_predictions": [...],
      "asst_predictions": [...]
    },
    ...
  },
  "per_conversation": [...]
}
```

## Utilities Reference

### `conversation_eval_utils.py`

Core functions:

- `load_conversations(path, limit)` - Load conversation data
- `extract_first_two_turns(conv)` - Get user/assistant text and emotions
- `get_turn_activation_avg_from_token(model, text, layer, start_token)` - Extract averaged activations
- `load_probe(path)` - Load probe from pickle
- `predict_emotion(activation, probe)` - Predict using PyTorch probe
- `load_model_and_tokenizer(model_name)` - Load model with proper configuration
- `load_probes_from_directory(dir, layers, pattern)` - Load multiple probes
- `evaluate_conversations(...)` - Main evaluation loop
- `save_results(results, path)` - Save JSON
- `print_results_summary(results)` - Print accuracy table

All functions are properly documented with type hints.

## Troubleshooting

### Import Error

Make sure you're in the right directory:
```bash
cd /workspace-vast/annas/git/research-tools
python probes/scripts/eval_probes_on_conversations.py --help
```

### Probe Not Found

Check probe files exist:
```bash
ls results/emotion_probes_high_alpha_cpca/probe_layer*.pkl
```

### Model Loading Issues

The utils use the working pre-load pattern:
1. Load with `AutoModelForCausalLM`
2. Wrap in `StandardizedTransformer`
3. Use `scan=False` in trace calls

This avoids the "meta device" bug.

## Extending

### Add New Probe Type

Edit `slurm_jobs/eval_all_probes_on_conversations.sh` and add:

```bash
echo "6/6: Testing my_probes..."
python probes/scripts/eval_probes_on_conversations.py \
    --conversations $CONVERSATIONS \
    --model $MODEL \
    --probe-dir results/my_probes \
    --probe-pattern "probe_layer{layer}_my_pattern.pkl" \
    --layers 10 20 30 \
    --output results/conversation_eval/gemma3_my_probes.json \
    --start-token $START_TOKEN \
    --limit $LIMIT \
    --probe-name "my_probes"
```

### Test Different Token Ranges

```bash
# First 10 tokens
--start-token 0 --limit-tokens 10  # (not implemented yet, but easy to add)

# Tokens 20-50
--start-token 20 --end-token 50  # (not implemented yet, but easy to add)
```

To implement, modify `get_turn_activation_avg_from_token` in utils.

## Job Status

**Current job:** 91146
- Status: Running
- Config: 200 conversations, 5 probe types
- ETA: ~30 minutes

Monitor:
```bash
squeue -j 91146
tail -f /workspace-vast/annas/logs/eval_conv_probes_91146.log
```
