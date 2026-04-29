#!/bin/bash
# Aggregate all 10-seed results (QCNN + re-uploading) once all four halves
# have finished. Run this in a 5th terminal after A/B scripts complete.
set -e
cd "$(dirname "$0")/.."
source venv/bin/activate

python - <<'PY'
import json, glob
from pathlib import Path
import numpy as np

LOGS = Path('results/logs')

def summarize_qcnn():
    accs_std, f1_std, mcc_std = [], [], []
    accs_zd,  f1_zd,  mcc_zd  = [], [], []
    seeds_found = []
    for seed in range(10):
        p = LOGS / f'qcnn_500s_seed{seed}.json'
        if not p.exists():
            print(f'  QCNN seed {seed}: MISSING')
            continue
        d = json.load(open(p))
        seeds_found.append(seed)
        accs_std.append(d['standard']['accuracy']); f1_std.append(d['standard']['f1']); mcc_std.append(d['standard']['mcc'])
        accs_zd.append(d['zeroday']['accuracy']);   f1_zd.append(d['zeroday']['f1']);   mcc_zd.append(d['zeroday']['mcc'])
        print(f'  seed {seed:2d} | std acc {d["standard"]["accuracy"]:.4f} | zd acc {d["zeroday"]["accuracy"]:.4f}')
    print()
    if accs_std:
        print(f'  STANDARD 3-class  | acc {np.mean(accs_std):.4f} ± {np.std(accs_std):.4f}  |  mcc {np.mean(mcc_std):.4f} ± {np.std(mcc_std):.4f}')
        print(f'  ZERO-DAY          | acc {np.mean(accs_zd):.4f}  ± {np.std(accs_zd):.4f}   |  mcc {np.mean(mcc_zd):.4f} ± {np.std(mcc_zd):.4f}')
    return {
        'seeds': seeds_found,
        'standard': {'acc_mean': float(np.mean(accs_std)), 'acc_std': float(np.std(accs_std)),
                     'mcc_mean': float(np.mean(mcc_std)), 'mcc_std': float(np.std(mcc_std))},
        'zeroday':  {'acc_mean': float(np.mean(accs_zd)),  'acc_std': float(np.std(accs_zd)),
                     'mcc_mean': float(np.mean(mcc_zd)),  'mcc_std': float(np.std(mcc_zd))},
    }

def summarize_reup():
    variants = ['single_qubit_standard', 'multi_qubit_standard', 'single_qubit_zeroday', 'multi_qubit_zeroday']
    labels   = ['1q standard', '4q standard', '1q zero-day', '4q zero-day']
    accs = {v: [] for v in variants}
    mccs = {v: [] for v in variants}
    for seed in range(10):
        p = LOGS / f'reuploading_500s_seed{seed}.json'
        if not p.exists():
            print(f'  Re-up seed {seed}: MISSING')
            continue
        d = json.load(open(p))
        for v in variants:
            accs[v].append(d[v]['accuracy']); mccs[v].append(d[v]['mcc'])
        print(f'  seed {seed:2d} | 1q std {d["single_qubit_standard"]["accuracy"]:.4f} | 4q std {d["multi_qubit_standard"]["accuracy"]:.4f} | 1q zd {d["single_qubit_zeroday"]["accuracy"]:.4f} | 4q zd {d["multi_qubit_zeroday"]["accuracy"]:.4f}')
    print()
    out = {}
    for v, lab in zip(variants, labels):
        if accs[v]:
            print(f'  {lab:14s} | acc {np.mean(accs[v]):.4f} ± {np.std(accs[v]):.4f}  |  mcc {np.mean(mccs[v]):.4f} ± {np.std(mccs[v]):.4f}')
            out[v] = {'acc_mean': float(np.mean(accs[v])), 'acc_std': float(np.std(accs[v])),
                      'mcc_mean': float(np.mean(mccs[v])), 'mcc_std': float(np.std(mccs[v]))}
    return out

print("=" * 60)
print("  QCNN — 10 seeds")
print("=" * 60)
qcnn = summarize_qcnn()

print()
print("=" * 60)
print("  DATA RE-UPLOADING — 10 seeds")
print("=" * 60)
reup = summarize_reup()

out = {'qcnn': qcnn, 'reuploading': reup}
with open(LOGS / 'quantum_10seed_summary.json', 'w') as f:
    json.dump(out, f, indent=2)
print()
print(f'Summary saved → {LOGS}/quantum_10seed_summary.json')
PY
