#!/bin/bash
# LEGACY 5-seed script (kept for historical runs).
# The report uses the 10-seed evaluation: see run_qcnn_10seed_A.sh and run_qcnn_10seed_B.sh.
# Run QCNN 500 samples with 5 different seeds (0-4)
cd "$(dirname "$0")/.."
source venv/bin/activate

for seed in 0 1 2 3 4; do
    echo ""
    echo "=========================================="
    echo "  QCNN 500 samples — SEED $seed"
    echo "=========================================="
    python3 -m src.quantum.qcnn --n_samples 500 --seed $seed
done

echo ""
echo "=========================================="
echo "  ALL SEEDS DONE — Summary"
echo "=========================================="
python3 -c "
import json, numpy as np
std_accs, zd_accs = [], []
std_f1s, zd_f1s = [], []
for seed in range(5):
    try:
        with open(f'results/logs/qcnn_500s_seed{seed}.json') as f:
            d = json.load(f)
        std_accs.append(d['standard']['accuracy'])
        std_f1s.append(d['standard']['f1'])
        zd_accs.append(d['zeroday']['accuracy'])
        zd_f1s.append(d['zeroday']['f1'])
        print(f'Seed {seed}: Standard Acc={d[\"standard\"][\"accuracy\"]:.4f} F1={d[\"standard\"][\"f1\"]:.4f} | Zero-day Acc={d[\"zeroday\"][\"accuracy\"]:.4f} F1={d[\"zeroday\"][\"f1\"]:.4f}')
    except Exception as e:
        print(f'Seed {seed}: MISSING ({e})')
print()
if std_accs:
    print(f'Standard — Acc: {np.mean(std_accs):.4f} +/- {np.std(std_accs):.4f} | F1: {np.mean(std_f1s):.4f} +/- {np.std(std_f1s):.4f}')
if zd_accs:
    print(f'Zero-day — Acc: {np.mean(zd_accs):.4f} +/- {np.std(zd_accs):.4f} | F1: {np.mean(zd_f1s):.4f} +/- {np.std(zd_f1s):.4f}')
"
