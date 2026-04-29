#!/bin/bash
# QCNN — seeds 5,6,7,8,9 (half B of 10-seed run)
# Run in its own terminal split. Pair with run_qcnn_10seed_A.sh.
set -e
cd "$(dirname "$0")/.."
source venv/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

mkdir -p results/logs
LOG=results/logs/qcnn_10seed_B.log
: > "$LOG"

for seed in 5 6 7 8 9; do
    echo "" | tee -a "$LOG"
    echo "=== QCNN — seed $seed — $(date +%H:%M:%S) ===" | tee -a "$LOG"
    python -m src.quantum.qcnn --n_samples 500 --seed $seed 2>&1 | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo "=== QCNN half-B DONE — $(date +%H:%M:%S) ===" | tee -a "$LOG"
