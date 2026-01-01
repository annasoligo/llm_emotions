# Setup Instructions for Emotional Context Evaluation System

## Prerequisites

You need a Python environment with the following packages installed:
- `anthropic` - For Claude API access
- `transformers` - Required by believe-it-or-not utils
- `tqdm` - For progress bars
- Standard library packages (asyncio, json, logging, etc.)

## Environment Setup Options

### Option 1: Use existing conda environment

If you have a conda environment with these packages:

```bash
conda activate your_env_name
export ANTHROPIC_API_KEY="your-api-key-here"
export PYTHONPATH=/workspace-vast/annas/git/believe-it-or-not:$PYTHONPATH
```

### Option 2: Create a new virtual environment

```bash
cd /workspace-vast/annas/git/research-tools/elicitation/emotional_context

# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Install requirements
pip install anthropic transformers tqdm

# Set environment variables
export ANTHROPIC_API_KEY="your-api-key-here"
export PYTHONPATH=/workspace-vast/annas/git/believe-it-or-not:$PYTHONPATH
```

### Option 3: Install to user directory (if allowed)

```bash
pip install --user anthropic transformers tqdm
export ANTHROPIC_API_KEY="your-api-key-here"
export PYTHONPATH=/workspace-vast/annas/git/believe-it-or-not:$PYTHONPATH
```

## Testing the Setup

Once your environment is configured, test it:

```bash
cd /workspace-vast/annas/git/research-tools/elicitation/emotional_context
python test_setup.py
```

You should see all tests passing:
```
✓ All local imports successful
✓ ANTHROPIC_API_KEY is set
✓ data/ exists
✓ logs/ exists
✓ outputs/ exists
✓ believe-it-or-not utilities accessible
✓ Anthropic client instantiated successfully
```

## Running with SLURM

### Create a SLURM script with your environment

Create `slurm_run_pipeline.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=emotional_context
#SBATCH --output=logs/pipeline_%j.log
#SBATCH --error=logs/pipeline_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

# Load your environment
source ~/.bashrc
conda activate your_env_name  # or: source venv/bin/activate

# Set environment variables
export ANTHROPIC_API_KEY="your-api-key-here"
export PYTHONPATH=/workspace-vast/annas/git/believe-it-or-not:$PYTHONPATH

# Run the pipeline
cd /workspace-vast/annas/git/research-tools/elicitation/emotional_context
python run_all_stages.py
```

Then submit:
```bash
sbatch slurm_run_pipeline.sh
```

### Or run individual stages

```bash
#!/bin/bash
#SBATCH --job-name=stage1_neutral_requests
#SBATCH --output=logs/stage1_%j.log
#SBATCH --error=logs/stage1_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

# Load your environment
source ~/.bashrc
conda activate your_env_name

# Set environment variables
export ANTHROPIC_API_KEY="your-api-key-here"
export PYTHONPATH=/workspace-vast/annas/git/believe-it-or-not:$PYTHONPATH

# Run stage 1
cd /workspace-vast/annas/git/research-tools/elicitation/emotional_context
python stage1_generate_neutral_requests.py
```

## Troubleshooting

### "No module named 'anthropic'"
Install the anthropic package in your environment:
```bash
pip install anthropic
```

### "No module named 'transformers'"
Install transformers:
```bash
pip install transformers
```

### "ANTHROPIC_API_KEY is not set"
Set the environment variable before running:
```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

### "believe-it-or-not access failed"
Ensure PYTHONPATH is set:
```bash
export PYTHONPATH=/workspace-vast/annas/git/believe-it-or-not:$PYTHONPATH
```

## Next Steps

After setup is complete and tests pass:

1. **Configure the pipeline** - Edit `config.py` if you want to change:
   - Number of requests (default: 100)
   - Number of responses per prefix (default: 10)
   - Models to use
   - Concurrency limits

2. **Run the pipeline** - Execute all stages:
   ```bash
   python run_all_stages.py
   ```

3. **Check outputs** - Results will be in the `data/` directory:
   - `stage6_final_dataset.jsonl` - Your final training dataset

## Estimated Resource Requirements

For default settings (100 base requests):
- **Time**: ~1-2 hours (depends on API rate limits)
- **Memory**: 8-16GB should be sufficient
- **API Calls**: ~4,800 total (100 + 400 + 4,000 + 4,000 generations/judgments)
- **Cost Estimate**: ~$50-100 depending on Claude pricing

## Support

If you encounter issues, check:
1. All imports work: `python test_setup.py`
2. API key is valid: Test with a simple Anthropic API call
3. Paths are correct: Check `config.py` paths
4. Logs for errors: Check `logs/` directory for detailed error messages
