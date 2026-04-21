#!/bin/bash
# Dataset size ablation study
# Runs quantum models with different training set sizes
# Usage: ./run_ablation.sh

set -e
cd "$(dirname "$0")"
source venv/bin/activate 2>/dev/null || true
export PYTHONUNBUFFERED=1

echo "============================================"
echo "  DATASET SIZE ABLATION STUDY"
echo "  Sizes: 100, 250, 1000"
echo "  (500 already done)"
echo "============================================"
echo ""

for N in 100 250 1000; do
    echo ""
    echo "########################################"
    echo "  STARTING n_samples = $N"
    echo "  $(date)"
    echo "########################################"
    echo ""

    echo ">>> Re-uploading (n=$N)..."
    python -m src.quantum.data_reuploading --n_samples $N

    echo ""
    echo ">>> QCNN (n=$N)..."
    python -m src.quantum.qcnn --n_samples $N

    echo ""
    echo ">>> DONE n_samples=$N at $(date)"
    echo ""
done

echo ""
echo "============================================"
echo "  ALL ABLATION RUNS COMPLETE"
echo "  $(date)"
echo "============================================"
echo ""
echo "Results saved in results/logs/:"
ls -la results/logs/*_*s.json
