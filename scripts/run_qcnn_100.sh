#!/bin/bash
# Run QCNN with 100 samples
cd "$(dirname "$0")"
source venv/bin/activate
echo "=== Running QCNN (100 samples) ==="
python3 -m src.quantum.qcnn --n_samples 100
echo "=== Done ==="
