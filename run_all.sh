#!/bin/bash
# Run everything in order
cd "$(dirname "$0")"
source venv/bin/activate

echo "=========================================="
echo "  1/9: Classical Baselines"
echo "=========================================="
python3 -m src.classical.classical_baselines

echo ""
echo "=========================================="
echo "  2/9: QCNN 100 samples"
echo "=========================================="
python3 -m src.quantum.qcnn --n_samples 100

echo ""
echo "=========================================="
echo "  3/9: QCNN 250 samples"
echo "=========================================="
python3 -m src.quantum.qcnn --n_samples 250

echo ""
echo "=========================================="
echo "  4/9: QCNN 500 samples"
echo "=========================================="
python3 -m src.quantum.qcnn --n_samples 500

echo ""
echo "=========================================="
echo "  5/9: QCNN 1000 samples"
echo "=========================================="
python3 -m src.quantum.qcnn --n_samples 1000

echo ""
echo "=========================================="
echo "  6/9: Re-uploading 100 samples"
echo "=========================================="
python3 -m src.quantum.data_reuploading --n_samples 100

echo ""
echo "=========================================="
echo "  7/9: Re-uploading 250 samples"
echo "=========================================="
python3 -m src.quantum.data_reuploading --n_samples 250

echo ""
echo "=========================================="
echo "  8/9: Re-uploading 500 samples"
echo "=========================================="
python3 -m src.quantum.data_reuploading --n_samples 500

echo ""
echo "=========================================="
echo "  9/9: Re-uploading 1000 samples"
echo "=========================================="
python3 -m src.quantum.data_reuploading --n_samples 1000

echo ""
echo "=========================================="
echo "  ALL DONE!"
echo "=========================================="
