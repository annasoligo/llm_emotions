---
name: slurm
description: Write and submit slurm job scripts for the RunPod shared cluster. Use when the user asks to run GPU jobs, submit experiments, create batch scripts, or debug slurm issues. Knows cluster-specific node exclusions, QoS strategy, storage tiers, and NCCL configuration.
---

# Slurm Skill

## Overview

This skill helps write correct slurm scripts for the RunPod shared GPU cluster. It handles node exclusions, QoS selection, NCCL configuration, cleanup traps, and storage tier management.

## Instructions

When the user invokes `/slurm` or asks to create/submit a slurm job:

### Step 1: Parse Arguments

The user may provide: `$ARGUMENTS`

Arguments can include:
- **Script path**: Python script to run
- **Model**: Which model to use (check CLAUDE.md models table)
- **GPUs**: Number of GPUs needed
- **QoS**: `high` or `low` (default: pick based on context)
- **Time**: Wall time limit
- **Other flags**: Any additional sbatch or script arguments

### Step 2: Determine Job Configuration

**GPU count** - check the models table in CLAUDE.md:

| Model | GPUs | TP Size | gpu_mem_util | max_model_len |
|-------|------|---------|-------------|---------------|
| Gemma 27B | 1 | 1 | 0.90 | 8192 |
| Gemma 12B | 1 | 1 | 0.90 | 8192 |
| Qwen 32B | 2 | 2 | 0.85 | 8192 |
| Qwen 235B | 4 | 4 | 0.80 | 4096 |
| Qwen 14B | 1 | 1 | 0.90 | 8192 |
| LLaMA 70B | 4 | 4 | 0.85 | 8192 |
| Mistral Nemo | 1 | 1 | 0.90 | 8192 |

**QoS selection**:
- `high` (default for single important jobs): ~12-15 GPU per-user quota, won't be preempted
- `low` (for sweeps or when at quota): no GPU limit, may be preempted
- `dev` (interactive `srun` ONLY, never with `sbatch`)

**Resource defaults by GPU count**:
- 1 GPU: `--cpus-per-task=8 --mem=48G`
- 2 GPUs: `--cpus-per-task=16 --mem=96G`
- 4 GPUs: `--cpus-per-task=32 --mem=192G`

### Step 3: Write the Slurm Script

Use this template, adapting as needed:

```bash
#!/bin/bash
#SBATCH --job-name=DESCRIPTIVE_NAME
#SBATCH --output=/workspace-vast/annas/logs/JOBNAME_%j.out
#SBATCH --error=/workspace-vast/annas/logs/JOBNAME_%j.out
#SBATCH --time=6:00:00
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:N
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --exclude=node-[0-1],node-10,node-12,node-[16-22]

# === Environment ===
source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

# === NCCL (required for multi-GPU) ===
export NCCL_SOCKET_IFNAME="=vxlan0"
export NCCL_P2P_DISABLE=1
export NCCL_NVLS_ENABLE=0

# === vLLM (if using vLLM) ===
export VLLM_USE_V1=0
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

# === Cleanup trap ===
cleanup() {
    echo "Cleaning up..."
    pkill -9 -f "vllm" || true
    pkill -9 -f "ray" || true
}
trap cleanup EXIT SIGTERM SIGINT

# === Run ===
python3 my_script.py --args
```

### Step 4: Submit

```bash
sbatch script_name.sh
```

Then check with: `squeue -u annas`

## Critical Rules

### ALWAYS include these in EVERY slurm script

1. **Secrets**: `source /workspace-vast/annas/.secrets/load_secrets.sh` (sets HF_HOME, API keys)
2. **Venv**: `source .venv/bin/activate`
3. **Node exclusions**: `#SBATCH --exclude=node-[0-1],node-10,node-12,node-[16-22]`
4. **Cleanup trap**: For vLLM/ray jobs, always add the cleanup trap

### NEVER do these

1. **NEVER** run GPU scripts directly - always `sbatch`
2. **NEVER** use `--qos=dev` with `sbatch` (dev is for interactive `srun` only)
3. **NEVER** manually set `CUDA_VISIBLE_DEVICES` (slurm handles this)
4. **NEVER** exceed `gpu_memory_utilization=0.80` for multi-GPU vLLM

## Cluster Architecture

### Nodes

```
node-0, node-1:   Controllers - NEVER run jobs here
node-2 to node-9: Compute nodes (primary workhorses)
node-10:          BROKEN (permission issues)
node-11:          Compute node
node-12:          BROKEN (permission issues)
node-13 to node-15: Compute nodes
node-16 to node-22: BROKEN (no /workspace/ mount)
```

### Storage Tiers

| Filesystem | Capacity | Speed | Use For |
|------------|----------|-------|---------|
| `/workspace-vast/` | 10TB | Fast (NVMe) | Code, final models, results, logs |
| `/workspace/` | 73TB | Slower (NFS) | Large temp files, training checkpoints |
| `/home/` | Small | Local overlay | **AVOID** - limited quota, causes "Disk quota exceeded" |

**CRITICAL**: Always set `HF_HOME=/workspace-vast/pretrained_ckpts` (done by `load_secrets.sh`) to prevent model downloads to `/home/`.

### QoS Details

| QoS | Priority | GPU Quota | Preemptible | Use When |
|-----|----------|-----------|-------------|----------|
| `high` | 200 | ~12-15 GPUs/user | No | Single important experiments |
| `low` | 100 | No limit | Yes | Sweeps, batch jobs, hitting quota |
| `dev` | 300 | Varies | No | Interactive `srun` ONLY |

**Check your quota**: `sacctmgr show qos format=name,MaxTRESPerUser%30`

**If blocked by QOSMaxGRESPerUser**: Switch to `--qos=low`, or cancel existing high jobs.

### Partitions

| Partition | Nodes | Purpose |
|-----------|-------|---------|
| `general` | 1-13 | Production batch jobs |
| `dev` | 14-15 | Interactive debugging |
| `overflow` | All | Flexible placement |

## NCCL Configuration

For multi-GPU jobs, ALWAYS set:

```bash
# The "=" prefix is NCCL syntax for "interfaces starting with vxlan0"
export NCCL_SOCKET_IFNAME="=vxlan0"
export NCCL_P2P_DISABLE=1
export NCCL_NVLS_ENABLE=0
```

Without these, multi-GPU jobs fail with `Bootstrap : no socket interface found`.

## Interactive Sessions

```bash
srun -p dev,overflow --qos=dev --cpus-per-task=8 --gres=gpu:1 --mem=32G --time=4:00:00 --job-name=D_annas --pty bash
```

- Use `D_` prefix: auto-cleaned at midnight PT
- Max 4 hours, 1-2 GPUs
- Use for debugging, not production runs

## SIGTERM Grace Period

For long jobs that save checkpoints, add:

```bash
#SBATCH --signal=B:SIGTERM@900
```

This sends SIGTERM 15 minutes before the time limit, allowing graceful checkpoint saves. Handle it in Python:

```python
import signal

def sigterm_handler(signum, frame):
    print("SIGTERM received - saving checkpoint...")
    # Save checkpoint logic
    sys.exit(0)

signal.signal(signal.SIGTERM, sigterm_handler)
```

**Note**: This only applies to time-limit termination. Preemption (low QoS) gives only ~3 minutes warning.

## Job Monitoring Commands

```bash
# Your jobs
squeue -u annas

# Detailed job info
scontrol show job <JOB_ID>

# Why is job pending?
scontrol show job <JOB_ID> | grep Reason

# Job history (completed/failed)
sacct -u annas --starttime=today --format=JobID,JobName,Elapsed,State,ExitCode

# Tail live logs
tail -f /workspace-vast/annas/logs/<name>_<job_id>.out

# Cancel job
scancel <JOB_ID>

# Cancel all my jobs
scancel -u annas
```

## Common Errors and Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `QOSMaxGRESPerUser` | Hit ~12-15 GPU quota on high QoS | Use `--qos=low` |
| `Bootstrap : no socket interface found` | Missing NCCL config | Add `NCCL_SOCKET_IFNAME="=vxlan0"` |
| `Permission denied: /workspace/` | Landed on broken node | Add `--exclude=node-[0-1],node-10,node-12,node-[16-22]` |
| `Disk quota exceeded` | Writing to `/home/` | Set `HF_HOME=/workspace-vast/pretrained_ckpts` |
| `No space left on device` | `/workspace/` or `/workspace-vast/` full | Clean old training dirs |
| Job killed without error | Non-slurm process killed by system | Must use `sbatch`, never run directly |
| `ModuleNotFoundError` | Venv not activated | Check `source .venv/bin/activate` in script |

## Example: vLLM Inference Job

```bash
#!/bin/bash
#SBATCH --job-name=steering_qwen235b
#SBATCH --output=/workspace-vast/annas/logs/steering_qwen235b_%j.out
#SBATCH --error=/workspace-vast/annas/logs/steering_qwen235b_%j.out
#SBATCH --time=6:00:00
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --exclude=node-[0-1],node-10,node-12,node-[16-22]

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

export VLLM_USE_V1=0
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export NCCL_SOCKET_IFNAME="=vxlan0"
export NCCL_P2P_DISABLE=1
export NCCL_NVLS_ENABLE=0

cleanup() {
    pkill -9 -f "vllm" || true
    pkill -9 -f "ray" || true
}
trap cleanup EXIT SIGTERM SIGINT

python3 steering_tests/vector_testing/behavioral_shift.py \
    --model Qwen/Qwen3-235B-A22B \
    --tensor-parallel 4 \
    --gpu-memory-utilization 0.80 \
    --max-model-len 4096
```

## Example: Lightweight Single-GPU Job

```bash
#!/bin/bash
#SBATCH --job-name=eval_gemma27b
#SBATCH --output=/workspace-vast/annas/logs/eval_gemma27b_%j.out
#SBATCH --error=/workspace-vast/annas/logs/eval_gemma27b_%j.out
#SBATCH --time=4:00:00
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --exclude=node-[0-1],node-10,node-12,node-[16-22]

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

python3 my_script.py --model google/gemma-3-27b-it
```

## Example: Array Job (Sweep)

For running the same script with different parameters:

```bash
#!/bin/bash
#SBATCH --job-name=sweep_emotions
#SBATCH --output=/workspace-vast/annas/logs/sweep_%A_%a.out
#SBATCH --error=/workspace-vast/annas/logs/sweep_%A_%a.out
#SBATCH --time=4:00:00
#SBATCH --partition=general
#SBATCH --qos=low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --array=0-14
#SBATCH --exclude=node-[0-1],node-10,node-12,node-[16-22]

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

EMOTIONS=(fear anxiety despair disgust guilt shame calm anger joy excitement hope curiosity interest pride frustration)
EMOTION=${EMOTIONS[$SLURM_ARRAY_TASK_ID]}

python3 my_script.py --emotion $EMOTION
```

Use `--qos=low` for sweeps to avoid hitting the ~12-15 GPU quota on high QoS.
