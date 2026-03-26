# Quantum ML for IoT Intrusion Detection

> Can quantum circuits detect network attacks that classical models have never seen?
> This project tries to find out.

**TER 2025–2026 — University of Strasbourg / ICube (UMR CNRS 7357)**
Supervisor: [Fabrice Théoleyre](mailto:fabrice.theoleyre@cnrs.fr)

---

## What this is

A head-to-head comparison of classical and quantum machine learning on a real-world IoT network dataset — 1 million flows, 3 traffic classes, and a zero-day detection scenario where the model is tested on attack types it has never seen during training.

Two quantum architectures are implemented from scratch using PennyLane:
- **QCNN** — Quantum Convolutional Neural Network (Hur et al., 2022)
- **Data Re-uploading** — universal single/multi-qubit classifier (Pérez-Salinas et al., 2020)

Both are benchmarked against SVM, Random Forest, and two neural networks under identical conditions.

---

## Dataset

`Network_dataset_11` from the ToN_IoT benchmark (UNSW Canberra):

| Class | Samples | Share |
|---|---|---|
| DoS attacks | 839,637 | 84.0% |
| Injection attacks | 125,195 | 12.5% |
| Normal traffic | 35,168 | 3.5% |

**Zero-day setup:** train on normal + DoS, evaluate on injection — which the model has never seen.

---

## Structure

```
├── src/
│   ├── preprocessing/      feature selection, normalization, balanced splits
│   ├── classical/          SVM, Random Forest, NN-Small, NN-Medium
│   └── quantum/            QCNN and data re-uploading (PennyLane)
├── notebooks/              step-by-step experiments
├── results/
│   ├── figures/            generated plots
│   └── logs/               metrics as JSON
├── report/                 full LaTeX manuscript
└── data/                   raw + preprocessed CSVs (not tracked)
```

---

## Quickstart

```bash
# 1. Set up environment
python3.11 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Run preprocessing (generates all CSV splits)
python src/preprocessing/preprocess.py

# 3. Classical baselines
python src/classical/classical_baselines.py

# 4. Build the report
cd report && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

---

## Status

| Component | Status |
|---|---|
| Preprocessing pipeline | done |
| Classical baselines | done |
| QCNN | in progress |
| Data Re-uploading | in progress |
| Results & report | in progress |

---

**Ashkan Motamedifar** — [montamedifar.ashkan@etu.unistra.fr](mailto:montamedifar.ashkan@etu.unistra.fr)
