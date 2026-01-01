#!/usr/bin/env python3
"""
Train K=50 orthogonal conversation probes (soft constraint) at layers 20-40.
"""
import subprocess
import sys

layers = list(range(20, 41))  # 20 to 40 inclusive
k_value = 50
ortho_weight = 100000.0

print("="*80)
print(f"Training K=50 Soft Constraint Conversation Probes")
print(f"Layers: {layers[0]}-{layers[-1]} ({len(layers)} layers)")
print("="*80)

for layer in layers:
    print(f"\n{'='*80}")
    print(f"Submitting job for layer {layer}")
    print(f"{'='*80}")
    
    cmd = [
        'sbatch',
        '--job-name', f'k50_l{layer}',
        '--output', f'/workspace-vast/annas/git/research-tools/probes/logs/train_k50_layer{layer}_%A.log',
        '--error', f'/workspace-vast/annas/git/research-tools/probes/logs/train_k50_layer{layer}_%A.err',
        '--time', '8:00:00',
        '--mem', '64G',
        '--cpus-per-task', '4',
        '--gres', 'gpu:1',
        '--wrap',
        f'cd /workspace-vast/annas/git/research-tools && '
        f'. .venv/bin/activate && '
        f'python probes/scripts/training/train_multi_orthogonal_conversation_probes.py '
        f'--layer {layer} '
        f'--n_sets {k_value} '
        f'--ortho_weight {ortho_weight} '
        f'--epochs 20 '
        f'--batch_size 32 '
        f'--lr 0.001'
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        job_id = result.stdout.strip().split()[-1]
        print(f"✓ Submitted job {job_id} for layer {layer}")
    else:
        print(f"✗ Failed to submit job for layer {layer}")
        print(f"Error: {result.stderr}")
        sys.exit(1)

print("\n" + "="*80)
print(f"✓ All {len(layers)} jobs submitted successfully!")
print("="*80)
