#!/bin/bash
# Run Data Re-uploading 500 samples with 5 different seeds
cd "$(dirname "$0")"
source venv/bin/activate

for seed in 0 1 2 3 4; do
    echo ""
    echo "=========================================="
    echo "  Re-uploading 500 samples — SEED $seed"
    echo "=========================================="
    python3 -m src.quantum.data_reuploading --n_samples 500 --seed $seed
done

echo ""
echo "=========================================="
echo "  ALL SEEDS DONE — Summary"
echo "=========================================="
python3 -c "
import json, numpy as np
models = ['single_qubit_standard', 'multi_qubit_standard', 'single_qubit_zeroday', 'multi_qubit_zeroday']
labels = ['1q Standard', '4q Standard', '1q Zero-day', '4q Zero-day']
accs = {m: [] for m in models}
f1s  = {m: [] for m in models}

for seed in range(5):
    try:
        with open(f'results/logs/reuploading_500s_seed{seed}.json') as f:
            d = json.load(f)
        line = f'Seed {seed}: '
        for m, l in zip(models, labels):
            accs[m].append(d[m]['accuracy'])
            f1s[m].append(d[m]['f1'])
            line += f'{l} Acc={d[m][\"accuracy\"]:.4f} | '
        print(line)
    except Exception as e:
        print(f'Seed {seed}: MISSING ({e})')
print()
for m, l in zip(models, labels):
    if accs[m]:
        print(f'{l:20s} — Acc: {np.mean(accs[m]):.4f} +/- {np.std(accs[m]):.4f} | F1: {np.mean(f1s[m]):.4f} +/- {np.std(f1s[m]):.4f}')
"
