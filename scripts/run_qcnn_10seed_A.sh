#!/bin/bash
# QCNN — seeds 0,1,2,3,4 (half A of 10-seed run)
# Run in its own terminal split. Pair with run_qcnn_10seed_B.sh.
set -e
cd "$(dirname "$0")/.."
source venv/bin/activate

# Keep each python process to one BLAS thread so 4 parallel scripts don't oversubscribe cores
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

mkdir -p results/logs
LOG=results/logs/qcnn_10seed_A.log
: > "$LOG"

for seed in 0 1 2 3 4; do
    echo "" | tee -a "$LOG"
    echo "=== QCNN — seed $seed — $(date +%H:%M:%S) ===" | tee -a "$LOG"
    python -m src.quantum.qcnn --n_samples 500 --seed $seed 2>&1 | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo "=== QCNN half-A DONE — $(date +%H:%M:%S) ===" | tee -a "$LOG"
