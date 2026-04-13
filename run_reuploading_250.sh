#!/bin/bash
# Run Data Re-uploading with 250 samples
cd "$(dirname "$0")"
source venv/bin/activate
echo "=== Running Data Re-uploading (250 samples) ==="
python3 -m src.quantum.data_reuploading --n_samples 250
echo "=== Done ==="
