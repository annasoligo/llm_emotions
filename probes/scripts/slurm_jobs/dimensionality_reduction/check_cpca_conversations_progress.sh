#!/bin/bash

echo "=== Conversation cPCA Progress Check ==="
echo ""

# Global cPCA
echo "--- Global cPCA (job 91151) ---"
if [ -d "/workspace-vast/annas/git/research-tools/probes/results/cpca_conversations_global/checkpoints" ]; then
    echo "Checkpoints:"
    ls -1 /workspace-vast/annas/git/research-tools/probes/results/cpca_conversations_global/checkpoints/ | head -5
    ckpt_count=$(ls -1 /workspace-vast/annas/git/research-tools/probes/results/cpca_conversations_global/checkpoints/ | wc -l)
    echo "Total checkpoints: $ckpt_count / 62 layers"
fi
echo ""

# Regional cPCA - User
echo "--- User Regional cPCA (job 91152) ---"
if [ -d "/workspace-vast/annas/git/research-tools/probes/results/cpca_conversations_regional_user/checkpoints" ]; then
    ckpt_count=$(ls -1 /workspace-vast/annas/git/research-tools/probes/results/cpca_conversations_regional_user/checkpoints/ 2>/dev/null | wc -l)
    echo "Total checkpoints: $ckpt_count / 62 layers"
else
    echo "Not started yet"
fi
echo ""

# Regional cPCA - Assistant
echo "--- Assistant Regional cPCA (job 91153) ---"
if [ -d "/workspace-vast/annas/git/research-tools/probes/results/cpca_conversations_regional_asst/checkpoints" ]; then
    ckpt_count=$(ls -1 /workspace-vast/annas/git/research-tools/probes/results/cpca_conversations_regional_asst/checkpoints/ 2>/dev/null | wc -l)
    echo "Total checkpoints: $ckpt_count / 62 layers"
else
    echo "Not started yet"
fi
echo ""

# Regional cPCA - Special1
echo "--- Special1 Regional cPCA (job 91154) ---"
if [ -d "/workspace-vast/annas/git/research-tools/probes/results/cpca_conversations_regional_special1/checkpoints" ]; then
    ckpt_count=$(ls -1 /workspace-vast/annas/git/research-tools/probes/results/cpca_conversations_regional_special1/checkpoints/ 2>/dev/null | wc -l)
    echo "Total checkpoints: $ckpt_count / 62 layers"
else
    echo "Not started yet"
fi
echo ""

# Regional cPCA - Special2
echo "--- Special2 Regional cPCA (job 91155) ---"
if [ -d "/workspace-vast/annas/git/research-tools/probes/results/cpca_conversations_regional_special2/checkpoints" ]; then
    ckpt_count=$(ls -1 /workspace-vast/annas/git/research-tools/probes/results/cpca_conversations_regional_special2/checkpoints/ 2>/dev/null | wc -l)
    echo "Total checkpoints: $ckpt_count / 62 layers"
else
    echo "Not started yet"
fi
echo ""

# Running jobs
echo "--- Running Jobs ---"
squeue -u annas | grep conv_cpca || squeue -u annas | grep conv_cpc
