---
name: slurm
description: Write and submit slurm job scripts for the RunPod shared cluster. Use when the user asks to run GPU jobs, submit experiments, create batch scripts, or debug slurm issues. Knows QoS strategy, storage tiers, and NCCL configuration.
---

# Slurm Skill

Helps write correct slurm scripts for the RunPod shared GPU cluster. For the canonical template, see `CLAUDE.md`. This skill covers decision logic, resource selection, and cluster-specific details.

## When writing a slurm script

### 1. Determine GPU count from the models table in CLAUDE.md

| Model | GPUs | TP Size | max_model_len |
|-------|------|---------|---------------|
| Gemma 27B | 1 | 1 | 8192 |
| Gemma 12B | 1 | 1 | 8192 |
| Qwen 32B | 2 | 2 | 8192 |
| Qwen 235B | 4 | 4 | 4096 |
| Qwen 14B | 1 | 1 | 8192 |
| LLaMA 70B | 4 | 4 | 8192 |
| Mistral Nemo | 1 | 1 | 8192 |

### 2. Set resources based on GPU count

| GPUs | cpus-per-task | mem |
|------|---------------|-----|
| 1 | 8 | 48G |
| 2 | 16 | 96G |
| 4 | 32 | 192G |

### 3. Choose QoS

| QoS | GPU Quota | Preemptible | Use When |
|-----|-----------|-------------|----------|
| `high` | ~12-15 GPUs/user | No | Single important experiments |
| `low` | No limit | Yes | Sweeps, batch jobs, hitting quota |
| `dev` | Varies | No | Interactive `srun` ONLY (never sbatch) |

**Check quota**: `sacctmgr show qos format=name,MaxTRESPerUser%30`
**If blocked by QOSMaxGRESPerUser**: Switch to `--qos=low`.

### 4. Use the template from CLAUDE.md

Always start from the template in CLAUDE.md. Key things that MUST be present:
1. `source /workspace-vast/annas/.secrets/load_secrets.sh`
2. `source .venv/bin/activate`
3. NCCL env vars for multi-GPU jobs
4. vLLM env vars if using vLLM
5. Cleanup trap for vLLM/ray jobs

### 5. Submit and verify

```bash
sbatch script_name.sh
squeue -u annas
```

## Storage Tiers

| Filesystem | Capacity | Speed | Use For |
|------------|----------|-------|---------|
| `/workspace-vast/` | 10TB | Fast (NVMe) | Code, final models, results, logs |
| `/workspace/` | 73TB | Slower (NFS) | Large temp files, training checkpoints |
| `/home/` | Small | Local overlay | **AVOID** - causes "Disk quota exceeded" |

## Interactive Sessions

```bash
srun -p dev,overflow --qos=dev --cpus-per-task=8 --gres=gpu:1 --mem=32G --time=4:00:00 --job-name=D_annas --pty bash
```

- `D_` prefix: auto-cleaned at midnight PT
- Max 4 hours, 1-2 GPUs
- For debugging only, not production runs

## SIGTERM Grace Period

For long jobs that save checkpoints:

```bash
#SBATCH --signal=B:SIGTERM@900
```

Sends SIGTERM 15 minutes before time limit. Only applies to time-limit termination; preemption (low QoS) gives only ~3 minutes.

## Job Monitoring

```bash
squeue -u annas                                    # Your jobs
scontrol show job <JOB_ID> | grep Reason           # Why pending?
sacct -u annas --starttime=today --format=JobID,JobName,Elapsed,State,ExitCode  # History
tail -f /workspace-vast/annas/logs/<name>_<id>.out  # Live logs
scancel <JOB_ID>                                   # Cancel job
```

## Common Errors

| Error | Fix |
|-------|-----|
| `QOSMaxGRESPerUser` | Use `--qos=low` |
| `Bootstrap : no socket interface found` | Add NCCL env vars (see CLAUDE.md template) |
| `Disk quota exceeded` | Ensure `load_secrets.sh` is sourced (sets HF_HOME) |
| Job killed silently | Must use `sbatch`, never run GPU scripts directly |
| `ModuleNotFoundError` | Check `source .venv/bin/activate` in script |
