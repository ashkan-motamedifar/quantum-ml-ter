#!/bin/bash
# Data Re-uploading (1q + 4q) — seeds 0,1,2,3,4 (half A of 10-seed run)
# Each seed trains BOTH the 1-qubit and 4-qubit variants.
# Run in its own terminal split. Pair with run_reup_10seed_B.sh.
set -e
cd "$(dirname "$0")/.."
source venv/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

mkdir -p results/logs
LOG=results/logs/reup_10seed_A.log
: > "$LOG"

for seed in 0 1 2 3 4; do
    echo "" | tee -a "$LOG"
    echo "=== Re-uploading — seed $seed — $(date +%H:%M:%S) ===" | tee -a "$LOG"
    python -m src.quantum.data_reuploading --n_samples 500 --seed $seed 2>&1 | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo "=== Re-uploading half-A DONE — $(date +%H:%M:%S) ===" | tee -a "$LOG"
