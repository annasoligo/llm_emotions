#!/bin/bash
# Check progress of orthogonal regularized probe training

echo "========================================"
echo "ORTHOGONAL REGULARIZED PROBE PROGRESS"
echo "========================================"
echo ""

# Check SLURM queue
echo "SLURM Jobs:"
echo "----------------------------------------"
squeue -u $USER -n ortho_test,neutral_pcs,ortho_reg_probe -o "%.10i %.20j %.8T %.10M %.10l %R" 2>/dev/null || echo "No jobs in queue"
echo ""

# Check test results
echo "Test Job Results:"
echo "----------------------------------------"
if [ -d "probes/results/orthogonal_regularized_probes_test" ]; then
    echo "✓ Test results directory exists"

    if [ -d "probes/results/orthogonal_regularized_probes_test/baseline" ]; then
        echo "✓ Baseline probe trained"
        if ls probes/results/orthogonal_regularized_probes_test/baseline/*_summary.txt 1> /dev/null 2>&1; then
            echo ""
            echo "Baseline results:"
            grep -E "Test accuracy|Neutral fraction" probes/results/orthogonal_regularized_probes_test/baseline/*_summary.txt 2>/dev/null | head -3
        fi
    else
        echo "  Baseline probe: not yet completed"
    fi

    if [ -d "probes/results/orthogonal_regularized_probes_test/lambda_1.0" ]; then
        echo "✓ Orthogonal probe trained"
        if ls probes/results/orthogonal_regularized_probes_test/lambda_1.0/*_summary.txt 1> /dev/null 2>&1; then
            echo ""
            echo "Orthogonal probe results:"
            grep -E "Test accuracy|Neutral fraction|Normalized overlap" probes/results/orthogonal_regularized_probes_test/lambda_1.0/*_summary.txt 2>/dev/null | head -5
        fi
    else
        echo "  Orthogonal probe: not yet completed"
    fi
else
    echo "  No test results yet"
fi
echo ""

# Check neutral PCs
echo "Neutral PCs:"
echo "----------------------------------------"
if [ -d "probes/results/neutral_pcs" ]; then
    N_PCS=$(find probes/results/neutral_pcs -name "layer_*.npy" 2>/dev/null | wc -l)
    echo "✓ $N_PCS layer(s) computed"

    if [ $N_PCS -gt 0 ]; then
        echo ""
        echo "Available layers:"
        ls probes/results/neutral_pcs/layer_*.npy 2>/dev/null | sed 's/.*layer_/  Layer /' | sed 's/_neutral_pcs_k/ (k=/' | sed 's/.npy/)/'
    fi

    if [ -f "probes/results/neutral_pcs/config_k20.json" ]; then
        echo ""
        echo "Configuration:"
        python3 -c "import json; data=json.load(open('probes/results/neutral_pcs/config_k20.json')); print(f\"  k={data['k']}, layers={data['layers']}\")" 2>/dev/null || echo "  (config exists but couldn't parse)"
    fi
else
    echo "  No neutral PCs computed yet"
fi
echo ""

# Check full sweep results
echo "Full Sweep Results:"
echo "----------------------------------------"
if [ -d "probes/results/orthogonal_regularized_probes" ]; then
    N_PROBES=$(find probes/results/orthogonal_regularized_probes -name "probe_*.pkl" 2>/dev/null | wc -l)
    echo "✓ $N_PROBES probe(s) trained"

    if [ $N_PROBES -gt 0 ]; then
        echo ""
        echo "Summary by layer and lambda:"
        echo "Layer | Lambda | Test Acc | Ortho Norm | Neutral Frac"
        echo "------|--------|----------|------------|-------------"

        for summary in probes/results/orthogonal_regularized_probes/*_summary.txt; do
            if [ -f "$summary" ]; then
                LAYER=$(grep "Layer:" "$summary" | awk '{print $2}')
                LAMBDA=$(grep "Lambda orthogonality:" "$summary" | awk '{print $3}')
                LAMBDA=${LAMBDA:-"baseline"}
                ACC=$(grep "Test accuracy:" "$summary" | awk '{print $3}')
                NORM=$(grep "Normalized overlap:" "$summary" | awk '{print $3}')
                FRAC=$(grep "Neutral fraction:" "$summary" | awk '{print $3}')

                if [ -n "$LAYER" ] && [ -n "$ACC" ]; then
                    printf "%5s | %6s | %8s | %10s | %12s\n" "$LAYER" "$LAMBDA" "$ACC" "${NORM:--}" "${FRAC:--}"
                fi
            fi
        done 2>/dev/null | sort -t'|' -k1 -k2
    fi
else
    echo "  No sweep results yet"
fi
echo ""

# Check recent log files
echo "Recent Log Activity:"
echo "----------------------------------------"
if [ -d "probes/logs" ]; then
    echo "Last 5 log files (most recent first):"
    ls -lt probes/logs/ortho*.log 2>/dev/null | head -5 | awk '{print "  " $9 " (" $6, $7, $8 ")"}'

    echo ""
    echo "Recent errors (if any):"
    grep -l "ERROR\|Error\|error" probes/logs/ortho*.err 2>/dev/null | head -3 | while read errfile; do
        echo "  $errfile:"
        tail -5 "$errfile" | sed 's/^/    /'
    done
else
    echo "  No logs directory found"
fi
echo ""

echo "========================================"
echo "To view detailed logs:"
echo "  tail -f probes/logs/ortho_test_*.log"
echo "  tail -f probes/logs/neutral_pcs_*.log"
echo "  tail -f probes/logs/ortho_reg_probe_*.log"
echo ""
echo "To cancel jobs:"
echo "  scancel <job_id>"
echo "========================================"
