#!/bin/bash
# Run Data Re-uploading with 500 samples
cd "$(dirname "$0")/.."
source venv/bin/activate
echo "=== Running Data Re-uploading (500 samples) ==="
python3 -m src.quantum.data_reuploading --n_samples 500
echo "=== Done ==="
