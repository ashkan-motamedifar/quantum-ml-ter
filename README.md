# Quantum ML for IoT Intrusion Detection

> Can quantum circuits detect network attacks that classical models have never seen?

**TER 2025-2026 — University of Strasbourg / ICube (UMR CNRS 7357)**
Supervisor: [Fabrice Théoleyre](mailto:fabrice.theoleyre@cnrs.fr)

---

## What this is

A head-to-head comparison of classical and quantum machine learning on the ToN_IoT dataset — 1 million IoT network flows, 3 traffic classes (normal, DoS, injection), and a zero-day scenario where models are tested on an attack type never seen during training.

Two quantum architectures are implemented from scratch in PennyLane:
- **QCNN** (Hur et al., 2022) — 8 qubits, 3 conv-pool stages
- **Data Re-uploading** (Pérez-Salinas et al., 2020) — single-qubit and 4-qubit variants

Benchmarked against SVM, Random Forest, and two neural networks under identical conditions, averaged over 5 seeds.

---

## Results summary

### Standard 3-class classification

| Model | Accuracy | MCC |
|---|---|---|
| Random Forest | 0.990 | 0.985 |
| SVM (RBF) | 0.970 | 0.955 |
| NN-Medium | 0.960 | 0.940 |
| NN-Small | 0.925 | 0.888 |
| Re-uploading 1q | 0.738 ± 0.024 | 0.645 ± 0.028 |
| Re-uploading 4q | 0.692 ± 0.005 | 0.586 ± 0.007 |
| QCNN (8q) | 0.677 ± 0.030 | 0.556 ± 0.051 |

### Zero-day detection (trained on normal+DoS, tested on injection)

| Model | Accuracy |
|---|---|
| QCNN (8q) — 5 seeds | 0.566 ± 0.281 |
| QCNN (8q) — 4 seeds (excl. barren plateau) | 0.705 ± 0.042 |
| Re-uploading 1q | 0.081 ± 0.020 |
| Re-uploading 4q | 0.070 ± 0.020 |
| Best classical (NN-Small) | 0.028 |

**Main finding:** the QCNN detects ~70% of unseen attacks on 4 out of 5 seeds, while every classical model fails below 3%. One seed collapses to 1% (barren plateau), illustrating the need for multi-seed evaluation.

---

## Dataset

`Network_dataset_11` from the ToN_IoT benchmark (UNSW Canberra):

| Class | Samples | Share |
|---|---|---|
| DoS | 839,637 | 84.0% |
| Injection | 125,195 | 12.5% |
| Normal | 35,168 | 3.5% |

Balanced via random undersampling to 35,168 per class.

---

## Structure

```
├── src/
│   ├── preprocessing/   feature selection, normalization, splits
│   ├── classical/       SVM, Random Forest, NN baselines
│   └── quantum/         QCNN, data re-uploading, validation
├── scripts/             shell scripts for running experiments
├── results/
│   ├── figures/         generated plots
│   └── logs/            metrics as JSON
├── report/              LaTeX manuscript
└── data/                raw + preprocessed CSVs (not tracked)
```

---

## Quickstart

```bash
# 1. Environment
python3.11 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Preprocess
python src/preprocessing/preprocess.py

# 3. Classical baselines
python src/classical/classical_baselines.py

# 4. Quantum classifiers (500 samples, 100 epochs, seed 42)
python -m src.quantum.qcnn
python -m src.quantum.data_reuploading

# 5. Multi-seed evaluation
bash scripts/run_qcnn_multiseed.sh
bash scripts/run_reuploading_multiseed.sh

# 6. Build report
cd report && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

---

**Ashkan Motamedifar** — [motamedifar.ashkan@etu.unistra.fr](mailto:motamedifar.ashkan@etu.unistra.fr)
