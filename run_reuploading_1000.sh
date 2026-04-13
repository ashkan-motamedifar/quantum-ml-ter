#!/bin/bash
# Run Data Re-uploading with 1000 samples
cd "$(dirname "$0")"
source venv/bin/activate
echo "=== Running Data Re-uploading (1000 samples) ==="
python3 -m src.quantum.data_reuploading --n_samples 1000
echo "=== Done ==="
