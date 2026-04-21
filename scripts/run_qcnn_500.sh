#!/bin/bash
# Run QCNN with 500 samples
cd "$(dirname "$0")"
source venv/bin/activate
echo "=== Running QCNN (500 samples) ==="
python3 -m src.quantum.qcnn --n_samples 500
echo "=== Done ==="
