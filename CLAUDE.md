# Claude Code Project Instructions

## Project Overview

Research tools for **emotion steering and interpretability in LLMs**. The core pipeline is:
1. **Activation collection** (`steering_tests/activation_collection/collect.py`) - collect hidden states from models
2. **Vector extraction** (`steering_tests/vector_extraction/extract_directions.py`) - compute steering directions (mean-diff)
3. **Behavioral testing** (`steering_tests/vector_testing/behavioral_shift.py`) - test behavioral impact via DOSPERT-style questions
4. **Analysis** (`steering_tests/analysis/`) - PCA, psychological axes projection, cross-model comparison

Secondary pipelines: frustration elicitation (`elicitation/`), emotion probes (`probes/`), comparative experiments (`experiments/`).

## Critical Rules (Common Failures)

### ALWAYS use slurm for GPU work
**NEVER** run GPU-intensive Python scripts directly. Always submit via `sbatch`. Non-slurm jobs get killed on this cluster.

### ALWAYS include secrets and venv in slurm scripts
Every slurm script MUST have this boilerplate:
```bash
source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate
```
This sets `HF_HOME=/workspace-vast/pretrained_ckpts` (avoids disk quota errors on `/home`) and loads API keys.

### ALWAYS save results progressively
**THIS IS EXTREMELY IMPORTANT.** For ANY long-running job (generation, judging, activation collection, behavioral testing), save results as they are produced. **NEVER** accumulate everything in memory and write at the end - if the job crashes at 95%, you lose everything. Use JSONL append, periodic JSON checkpoint dumps, or write-per-batch. Every result should be on disk within seconds of being generated.

### NEVER overwrite existing result files without asking
When rerunning experiments, write to new files (e.g., with updated timestamps) unless explicitly told to overwrite. Ask before deleting data.

### Commit frequently
Proactively suggest commits at natural breakpoints: after finishing a script, before submitting a slurm job, after getting a plot working, before starting a refactor. Aim for 3-8 commits per working session rather than one big commit at the end. This is critical for reproducibility - we need to know exactly what code produced which results.

### Copy existing code patterns, don't rewrite from scratch
When implementing something similar to existing code, find the existing implementation first and adapt it. Check `steering_tests/`, `elicitation/`, and `experiments/` for patterns.

### NEVER code in fallbacks or default values for data/models
**THIS IS EXTREMELY IMPORTANT.** If data cannot be loaded, a model fails to initialize, a file is missing, or there is any mismatch between expected and actual inputs, the code MUST raise an error and fail loudly. **NEVER** silently substitute defaults, fall back to alternative values, use empty data, skip missing items, or continue with partial results. Silent fallbacks hide bugs and produce garbage results that waste GPU hours. If something is wrong, crash immediately with a clear error message.

### ALWAYS sanity-check outputs before saving
Given our history of normalisation bugs (21x!), always print summary stats (min, max, mean, shape, count) when producing numerical outputs. Z-scores should be centered near 0; percentages should be 0-100; scores on a 0-10 scale should actually be 0-10. If values look wrong, fail rather than saving garbage.

## Environment

- **Python**: 3.12 in `.venv` (managed with `uv`)
- **Package manager**: `uv` (use `uv pip install` not plain `pip`)
- **Key packages**: torch, transformers, vllm, anthropic, numpy, matplotlib, seaborn
- **Logs**: `/workspace-vast/annas/logs/`
- **Model cache**: `/workspace-vast/pretrained_ckpts` (via HF_HOME)

## Models

| Model | HF Path | GPUs | TP Size | max_model_len |
|-------|---------|------|---------|---------------|
| Gemma 27B | `google/gemma-3-27b-it` | 1 | 1 | 8192 |
| Gemma 12B | `google/gemma-3-12b-it` | 1 | 1 | 8192 |
| Qwen 32B | `Qwen/Qwen3-32B` | 2 | 2 | 8192 |
| Qwen 235B | `Qwen/Qwen3-235B-A22B` | 4 | 4 | 4096 |
| Qwen 14B | `Qwen/Qwen3-14B` | 1 | 1 | 8192 |
| LLaMA 70B | `meta-llama/Llama-3.3-70B-Instruct` | 4 | 4 | 8192 |
| Mistral Nemo | `mistralai/Mistral-Nemo-Instruct-2407` | 1 | 1 | 8192 |

**IMPORTANT**:
- **ALWAYS use Qwen 3 and Gemma 3** (not older versions). If you see Qwen2 or Gemma 2 references in old code, update them.
- Gemma 3 has 62 layers (not Gemma 2). Don't confuse model versions.
- For Qwen3 models, disable thinking mode with `/no_think` suffix or `enable_thinking=False`.

## vLLM Configuration

See `.claude/skills/vllm.md` for full details. Key points:
- **ALWAYS** set `enforce_eager=True` (required for steering hooks)
- **ALWAYS** set `gpu_memory_utilization=0.80` (never higher - OOM risk during KV cache allocation)
- **ALWAYS** set these env vars in slurm scripts for multi-GPU:
  ```bash
  export VLLM_USE_V1=0
  export VLLM_ALLOW_INSECURE_SERIALIZATION=1
  export NCCL_P2P_DISABLE=1
  export NCCL_SOCKET_IFNAME="=vxlan0"
  export NCCL_NVLS_ENABLE=0
  ```
- **ALWAYS** add cleanup traps to kill orphaned vLLM/ray processes on exit

## API Backends

### Claude Models
Default model for data generation and judging: **`claude-sonnet-4-5-20250929`** (Sonnet 4.5).
- If user requests Haiku: use **`claude-haiku-4-5-20251001`**
- If user requests Opus: use **`claude-opus-4-6`**
- **NEVER** use older Claude model IDs (claude-3-*, claude-3.5-*, etc.)

### API Concurrency
- **Anthropic API**: Default 50 concurrent requests (`asyncio.Semaphore(50)`)
- **OpenRouter API**: Default 50 concurrent requests
- **CRITICAL**: These limits are **per-API-key across ALL running jobs**, not per-job. If you have 2 jobs running that both hit the Anthropic API, each should use `Semaphore(25)` so they share the 50 total. Always consider what else might be running when setting concurrency.

### Inference Backends
- **Anthropic**: Used for judging (Claude Sonnet 4.5). Backend in `elicitation/inference/anthropic_backend.py`.
- **OpenRouter**: Used for closed model evals (Gemini, GPT, etc.). 5 retries with exponential backoff. Backend in `elicitation/inference/openrouter.py`.

## Plotting Style

**ALWAYS check existing plots and config files for the correct colour scheme before making new plots.** This is a very frequent source of rework.

- Library: `matplotlib` with `seaborn-v0_8-whitegrid` style
- Colour palette (muted): see `steering_tests/vector_testing/config.py:VECTOR_COLORS`
- Category colours:
  ```python
  "#D4876A"  # Coral/Terra Cotta
  "#7BA7D7"  # Sky Blue
  "#7D9B7D"  # Olive Green
  "#C17B8D"  # Dusty Rose/Pink
  "#B8CCC8"  # Sage Green
  ```
- `figsize=(12, 8)`, `dpi=150`, `bbox_inches='tight'`
- Error bars: 90% confidence intervals unless otherwise specified
- No N= labels on plots unless requested

## Cluster Architecture

### Storage Tiers
| Filesystem | Capacity | Speed | Use For |
|------------|----------|-------|---------|
| `/workspace-vast/` | 10TB | Fast (NVMe) | Code, results, final models, logs |
| `/workspace/` | 73TB | Slower (NFS) | Large temp files, training checkpoints |
| `/home/` | Small | Local overlay | **AVOID** - causes "Disk quota exceeded" |

### QoS Strategy
- `high` (default): ~12-15 GPU quota/user, not preempted. Use for single important jobs.
- `low`: No GPU limit, can be preempted. Use for sweeps or when hitting quota.
- `dev`: Interactive `srun` ONLY. Never with `sbatch`.
- If blocked by `QOSMaxGRESPerUser`: switch to `--qos=low`.

See `.claude/skills/slurm/SKILL.md` for full cluster details.

## Slurm Script Template

```bash
#!/bin/bash
#SBATCH --job-name=descriptive_name
#SBATCH --output=/workspace-vast/annas/logs/jobname_%j.out
#SBATCH --error=/workspace-vast/annas/logs/jobname_%j.out
#SBATCH --time=6:00:00
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:N
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

# For multi-GPU jobs (NCCL):
export NCCL_SOCKET_IFNAME="=vxlan0"
export NCCL_P2P_DISABLE=1
export NCCL_NVLS_ENABLE=0

# For vLLM jobs:
export VLLM_USE_V1=0
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

# Cleanup trap
cleanup() {
    pkill -9 -f "vllm" || true
    pkill -9 -f "ray" || true
}
trap cleanup EXIT SIGTERM SIGINT

python3 my_script.py --args
```

## Directory Structure

```
steering_tests/           # Core steering vector pipeline
  activation_collection/  # Collect hidden states (collect.py)
  vector_extraction/      # Extract directions (extract_directions.py)
  vector_testing/         # Behavioral tests (behavioral_shift.py, config.py)
  analysis/               # PCA, psychological axes plots
  steering_utils/         # VLLMSteering class, layer norms, plotting
  data/                   # Emotion prompts, text pairs, appraisal scenarios
  activations/            # Collected activation pickle files
  vectors/                # Extracted steering vectors
elicitation/              # Frustration elicitation & generalization evals
  inference/              # OpenRouter and Anthropic API backends
  prompts/                # Judge prompts, eval prompts
experiments/              # Comparative experiments (base vs instruct, etc.)
  prefill_scaled/         # Prefill-based experiments
  steering/               # Applied steering experiments
probes/                   # Emotion probe training (orthogonal, cPCA)
  core/                   # Activation loading, data processing
  methods/                # Probe training (probes.py, cpca.py)
```

## Key Config Files

- `steering_tests/vector_testing/config.py` - Vector types, emotions, scales, colours, predictions
- `steering_tests/behavioral_experiments/config.py` - Model-specific configs (layers, hidden dims, stop tokens)
- `elicitation/prompts/judges.py` - `get_negativity_judge_prompt()` for frustration rating (0-10 scale)

## Output Conventions

- **Timestamps**: `%Y%m%d_%H%M%S` (e.g., `20260207_103722`)
- **File naming**: `{description}_{model_short}_{scenario}_{timestamp}.{ext}`
- **Activations**: `layer_00.pkl`, `layer_01.pkl` + `metadata.json` sidecar
- **Vectors**: `{emotion}.npy` in `vectors/{model}/{method}/` directories
- **Results**: JSONL for streaming, JSON for full results
- **Plots**: PNG at 150 DPI, saved alongside data or in `plots/` directory
