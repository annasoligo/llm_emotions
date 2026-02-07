# vLLM Best Practices for Slurm Jobs

## Environment Variables (Required)

Always set these environment variables in slurm scripts before running vLLM:

```bash
# Disable vLLM V1 engine (causes serialization issues with steering hooks)
export VLLM_USE_V1=0

# Allow serialization for steering vectors
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

# Disable P2P to avoid NCCL issues on some nodes
export NCCL_P2P_DISABLE=1

# CRITICAL: Set network interface for NCCL communication
# The "=" prefix is NCCL syntax for "interfaces starting with vxlan0"
export NCCL_SOCKET_IFNAME="=vxlan0"

# Disable NVLS to avoid errors on some nodes
export NCCL_NVLS_ENABLE=0

# Optional: Enable NCCL debug logging for troubleshooting
export NCCL_DEBUG=WARN
```

## GPU Memory Configuration

**IMPORTANT:** Never exceed `gpu_memory_utilization=0.80` to avoid OOM errors during KV cache allocation.

```python
llm = LLM(
    model=model_name,
    tensor_parallel_size=4,  # Match #SBATCH --gres=gpu:N
    gpu_memory_utilization=0.75,  # Safe default, max 0.80
    max_model_len=4096,  # Reduce for large models to save KV cache memory
    enforce_eager=True,  # Required for steering hooks
    disable_log_stats=True,
)
```

### Memory Guidelines by Model Size

| Model | GPUs | gpu_memory_utilization | max_model_len |
|-------|------|------------------------|---------------|
| <30B  | 2    | 0.80                   | 8192          |
| 30-70B| 4    | 0.75                   | 4096-8192     |
| >100B | 4-8  | 0.75                   | 4096          |

## Cleanup and Orphan Prevention

### In Python Scripts

Always wrap vLLM usage in try/finally to ensure cleanup:

```python
import atexit
import signal

def cleanup():
    """Clean up vLLM resources."""
    if 'steering' in dir():
        steering.clear()
    if 'llm' in dir():
        del llm
    import gc
    gc.collect()
    import torch
    torch.cuda.empty_cache()

atexit.register(cleanup)

# Handle SIGTERM from slurm
def sigterm_handler(signum, frame):
    cleanup()
    sys.exit(0)

signal.signal(signal.SIGTERM, sigterm_handler)
```

### In Slurm Scripts

Add cleanup trap at the end of slurm scripts:

```bash
# At the start of script
cleanup() {
    echo "Cleaning up..."
    # Kill any orphaned python processes from this job
    pkill -9 -f "vllm" || true
    pkill -9 -f "ray" || true
}
trap cleanup EXIT SIGTERM SIGINT
```

## Common Errors and Solutions

### "NCCL error: invalid usage"
- **Cause:** NCCL can't find the right network interface
- **Fix:** Add `export NCCL_SOCKET_IFNAME=vxlan0`

### "No available memory for cache blocks"
- **Cause:** Model weights use too much GPU memory
- **Fix:** Reduce `max_model_len` (e.g., 8192 → 4096) or `gpu_memory_utilization`

### "Engine core initialization failed"
- **Cause:** Often a combination of memory and NCCL issues
- **Fix:** Apply all environment variables above and reduce memory usage

### Jobs fail quickly (<1 min) on some nodes
- **Cause:** Some nodes have other processes using GPU memory
- **Fix:** Resubmit - jobs will land on different nodes. Or use `--exclusive` flag (but this limits scheduling)

## Example Slurm Script Header

```bash
#!/bin/bash
#SBATCH --job-name=my_vllm_job
#SBATCH --output=/workspace-vast/annas/logs/my_job_%A_%a.out
#SBATCH --error=/workspace-vast/annas/logs/my_job_%A_%a.out
#SBATCH --time=6:00:00
#SBATCH --partition=general
#SBATCH --gres=gpu:4

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

# vLLM environment setup
export VLLM_USE_V1=0
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME="=vxlan0"
export NCCL_NVLS_ENABLE=0

# Cleanup trap
cleanup() {
    pkill -9 -f "vllm" || true
    pkill -9 -f "ray" || true
}
trap cleanup EXIT SIGTERM SIGINT

# Your code here...
```
