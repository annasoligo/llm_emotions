# vLLM Best Practices

This covers vLLM-specific configuration. For slurm templates and cluster details, see `CLAUDE.md`.

## Environment Variables (Required)

**IMPORTANT:** `VLLM_USE_V1=0` and `VLLM_ALLOW_INSECURE_SERIALIZATION=1` are required for **ALL** vLLM jobs — including single-GPU. Without them, steering hooks fail with `TypeError: Object of type _SetupMultiLayerHookCallable is not serializable`. This is the #1 cause of "Engine core died unexpectedly" errors.

```bash
# Required for ALL vLLM steering jobs (single-GPU and multi-GPU)
export VLLM_USE_V1=0
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

# NCCL networking (required for multi-GPU only)
# The "=" prefix is NCCL syntax for "interfaces starting with vxlan0"
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME="=vxlan0"
export NCCL_NVLS_ENABLE=0
```

## GPU Memory Configuration

Use `gpu_memory_utilization=0.80` for most models. **Exception:** Qwen 235B (109.5 GiB across 4 GPUs) needs `gpu_memory_utilization=0.90` — at 0.80 there is negative KV cache headroom and vLLM will fail with "No available memory for cache blocks".

```python
# Most models
llm = LLM(
    model=model_name,
    tensor_parallel_size=4,  # Match #SBATCH --gres=gpu:N
    gpu_memory_utilization=0.80,
    max_model_len=8192,
    enforce_eager=True,  # Required for steering hooks
    disable_log_stats=True,
)

# Qwen 235B specifically
llm = LLM(
    model="Qwen/Qwen3-235B-A22B",
    tensor_parallel_size=4,
    gpu_memory_utilization=0.90,  # 0.80 causes OOM - model is 109.5 GiB
    max_model_len=8192,
    enforce_eager=True,
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
- **Cause:** Model weights use too much GPU memory for the `gpu_memory_utilization` budget
- **Fix:** For Qwen 235B, use `gpu_memory_utilization=0.90` (not 0.80). For other models, reduce `max_model_len` or check that no other processes are using GPU memory on the node (use `--exclude=node-X` to avoid busy nodes)

### "Object of type _SetupMultiLayerHookCallable is not serializable" / "Engine core died unexpectedly"
- **Cause:** Missing `VLLM_ALLOW_INSECURE_SERIALIZATION=1`. This is needed for ALL vLLM jobs (including single-GPU) when using steering hooks.
- **Fix:** Add `export VLLM_ALLOW_INSECURE_SERIALIZATION=1` to the slurm script

### "Engine core initialization failed"
- **Cause:** Often a combination of memory and NCCL issues
- **Fix:** Apply all environment variables above and reduce memory usage

### Jobs fail quickly (<1 min) on some nodes
- **Cause:** Some nodes have other processes using GPU memory
- **Fix:** Resubmit - jobs will land on different nodes
