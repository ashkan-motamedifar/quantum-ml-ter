#!/bin/bash
# Run QCNN with 250 samples
cd "$(dirname "$0")"
source venv/bin/activate
echo "=== Running QCNN (250 samples) ==="
python3 -m src.quantum.qcnn --n_samples 250
echo "=== Done ==="
