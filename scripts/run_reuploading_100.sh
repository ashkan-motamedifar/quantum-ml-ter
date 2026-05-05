#!/bin/bash
# Run Data Re-uploading with 100 samples
cd "$(dirname "$0")/.."
source venv/bin/activate
echo "=== Running Data Re-uploading (100 samples) ==="
python3 -m src.quantum.data_reuploading --n_samples 100
echo "=== Done ==="
