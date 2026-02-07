# vLLM Best Practices

This covers vLLM-specific configuration. For slurm templates and cluster details, see `CLAUDE.md`.

## Environment Variables (Required)

Always set these in slurm scripts before running vLLM:

```bash
# Disable vLLM V1 engine (causes serialization issues with steering hooks)
export VLLM_USE_V1=0

# Allow serialization for steering vectors
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

# NCCL networking (required for multi-GPU)
# The "=" prefix is NCCL syntax for "interfaces starting with vxlan0"
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME="=vxlan0"
export NCCL_NVLS_ENABLE=0
```

## GPU Memory Configuration

**ALWAYS use `gpu_memory_utilization=0.80`.** Never higher - risks OOM during KV cache allocation.

```python
llm = LLM(
    model=model_name,
    tensor_parallel_size=4,  # Match #SBATCH --gres=gpu:N
    gpu_memory_utilization=0.80,
    max_model_len=4096,  # Reduce for large models to save KV cache memory
    enforce_eager=True,  # Required for steering hooks
    disable_log_stats=True,
)
```

For model-specific GPU counts and max_model_len, see the models table in `CLAUDE.md`.

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

Always add a cleanup trap (see template in `CLAUDE.md`):

```bash
cleanup() {
    echo "Cleaning up..."
    pkill -9 -f "vllm" || true
    pkill -9 -f "ray" || true
}
trap cleanup EXIT SIGTERM SIGINT
```

## Common Errors and Solutions

### "NCCL error: invalid usage" / "Bootstrap : no socket interface found"
- **Cause:** NCCL can't find the right network interface
- **Fix:** Add `export NCCL_SOCKET_IFNAME="=vxlan0"` and `export NCCL_NVLS_ENABLE=0`

### "No available memory for cache blocks"
- **Cause:** Model weights use too much GPU memory
- **Fix:** Reduce `max_model_len` (e.g., 8192 -> 4096) or ensure `gpu_memory_utilization=0.80`

### "Engine core initialization failed"
- **Cause:** Often a combination of memory and NCCL issues
- **Fix:** Apply all environment variables above and reduce memory usage

### Jobs fail quickly (<1 min) on some nodes
- **Cause:** Some nodes have other processes using GPU memory
- **Fix:** Resubmit - jobs will land on different nodes
