#!/bin/bash
# Run QCNN with 1000 samples
cd "$(dirname "$0")/.."
source venv/bin/activate
echo "=== Running QCNN (1000 samples) ==="
python3 -m src.quantum.qcnn --n_samples 1000
echo "=== Done ==="
