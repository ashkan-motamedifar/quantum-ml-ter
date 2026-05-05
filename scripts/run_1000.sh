#!/bin/bash
# Run ONLY n=1000 with mini-batch (fast)
cd "$(dirname "$0")/.."
source venv/bin/activate 2>/dev/null || true
export PYTHONUNBUFFERED=1

echo "=========================================="
echo "  n=1000 with mini-batch (batch_size=100)"
echo "  $(date)"
echo "=========================================="

echo ">>> Re-uploading (n=1000)..."
python -m src.quantum.data_reuploading --n_samples 1000

echo ""
echo ">>> QCNN (n=1000)..."
python -m src.quantum.qcnn --n_samples 1000

echo ""
echo "=========================================="
echo "  DONE at $(date)"
echo "=========================================="
ls -la results/logs/*1000*
