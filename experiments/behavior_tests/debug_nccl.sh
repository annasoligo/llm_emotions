#!/bin/bash
#SBATCH --job-name=debug_nccl
#SBATCH --output=/workspace-vast/annas/logs/debug_nccl_%j.out
#SBATCH --error=/workspace-vast/annas/logs/debug_nccl_%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0:10:00

# Debug NCCL issues on this node

echo "=== NCCL Debug Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
date

echo ""
echo "=== GPU Info ==="
nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.free --format=csv

echo ""
echo "=== GPU Topology ==="
nvidia-smi topo -m

echo ""
echo "=== NCCL Environment ==="
env | grep -i nccl

echo ""
echo "=== Testing NCCL with torch ==="
source /workspace-vast/annas/git/research-tools/.venv/bin/activate

python3 << 'EOF'
import os
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_DEBUG_SUBSYS"] = "ALL"

import torch
import torch.distributed as dist

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")

for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"GPU {i}: {props.name}, {props.total_memory / 1e9:.1f} GB")

# Try a simple NCCL operation
print("\nTesting NCCL initialization...")
try:
    # Simple test - create tensors on each GPU
    tensors = []
    for i in range(min(4, torch.cuda.device_count())):
        with torch.cuda.device(i):
            t = torch.ones(1000, device=f'cuda:{i}')
            tensors.append(t)
    print(f"Created tensors on {len(tensors)} GPUs")

    # Test P2P access
    print("\nP2P Access Matrix:")
    n_gpus = min(4, torch.cuda.device_count())
    for i in range(n_gpus):
        row = []
        for j in range(n_gpus):
            if i == j:
                row.append("X")
            else:
                can_access = torch.cuda.can_device_access_peer(i, j)
                row.append("Y" if can_access else "N")
        print(f"  GPU {i}: {' '.join(row)}")

    print("\nNCCL basic test passed!")
except Exception as e:
    print(f"NCCL test failed: {e}")
    import traceback
    traceback.print_exc()

EOF

echo ""
echo "=== Done ==="
date
