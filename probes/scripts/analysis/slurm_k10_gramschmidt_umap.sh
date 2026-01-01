#!/bin/bash
#SBATCH --job-name=k10_gs_umap
#SBATCH --output=/workspace-vast/annas/git/research-tools/probes/logs/k10_gramschmidt_umap_%A.log
#SBATCH --error=/workspace-vast/annas/git/research-tools/probes/logs/k10_gramschmidt_umap_%A.err
#SBATCH --time=1:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# UMAP analysis on K=10 Gram-Schmidt probes

cd /workspace-vast/annas/git/research-tools

echo "========================================"
echo "K=10 Gram-Schmidt UMAP Analysis"
echo "========================================"
echo ""

# Use system python3 with user-installed packages (not venv)
python3 probes/scripts/analysis/run_k10_gramschmidt_umap.py

EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: UMAP analysis failed"
    exit 1
fi

echo ""
echo "========================================"
echo "✓ K=10 Gram-Schmidt UMAP completed!"
echo "========================================"
