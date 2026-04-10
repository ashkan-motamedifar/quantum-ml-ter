# Quantum ML for IoT Intrusion Detection

> Can quantum circuits detect network attacks that classical models have never seen?
> This project tries to find out.

**TER 2025-2026 -- University of Strasbourg / ICube (UMR CNRS 7357)**
Supervisor: [Fabrice Theoleyre](mailto:fabrice.theoleyre@cnrs.fr)

---

## What this is

A head-to-head comparison of classical and quantum machine learning on a real-world IoT network dataset -- 1 million flows, 3 traffic classes, and a zero-day detection scenario where the model is tested on attack types it has never seen during training.

Two quantum architectures are implemented from scratch using PennyLane:
- **QCNN** -- Quantum Convolutional Neural Network (Hur et al., 2022)
- **Data Re-uploading** -- universal single/multi-qubit classifier (Perez-Salinas et al., 2020)

Both are benchmarked against SVM, Random Forest, and two neural networks under identical conditions.

**Key finding:** Classical models reach 99% on standard classification, but the QCNN detects 91-98% of unseen attack types in zero-day detection -- while every classical model fails below 3%. Entanglement is essential: the single-qubit model (no entanglement) fails at 0%.

---

## Results summary

### Standard 3-class classification (100 epochs)

| Model | Accuracy | MCC |
|---|---|---|
| SVM / RF / NN-Small | 99.0% | 0.985 |
| Re-uploading 1q | 72.5% | 0.625 |
| Re-uploading 4q | 71.5% | 0.638 |
| QCNN (8q) | 70.5% | 0.612 |

### Zero-day detection (trained on normal+DoS, tested on injection)

| Model | Recall |
|---|---|
| QCNN (8q) | 66-98% (seed-dependent) |
| Re-uploading 4q | 0-85% (seed-dependent) |
| Classical models | < 2.3% |

### Ablation studies
- **Epochs (30 vs 100):** Quantum models improve +9 to +34.5pp but plateau at ~70-75%
- **Dataset size (100 to 1000):** Minimal effect, confirming circuit capacity is the bottleneck

---

## Dataset

`Network_dataset_11` from the ToN_IoT benchmark (UNSW Canberra):

| Class | Samples | Share |
|---|---|---|
| DoS attacks | 839,637 | 84.0% |
| Injection attacks | 125,195 | 12.5% |
| Normal traffic | 35,168 | 3.5% |

**Zero-day setup:** train on normal + DoS, evaluate on injection (never seen during training).

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
│   └── logs/               metrics as JSON (100s, 250s, 500s, 1000s ablations)
├── report/                 full LaTeX manuscript (34 pages)
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

# 4. Quantum classifiers (default: 500 samples, 100 epochs)
python -m src.quantum.qcnn
python -m src.quantum.data_reuploading

# 5. Dataset size ablation
python -m src.quantum.qcnn --n_samples 1000
python -m src.quantum.data_reuploading --n_samples 1000

# 6. Build the report
cd report && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

---

## Status

| Component | Status |
|---|---|
| Preprocessing pipeline | done |
| Classical baselines | done |
| QCNN | done |
| Data Re-uploading | done |
| Validation (synthetic datasets) | done |
| Ablation: epochs (30 vs 100) | done |
| Ablation: dataset size (100-1000) | done |
| ROC analysis | done |
| Report (Ch1-7 + conclusion) | done |

---

**Ashkan Motamedifar** -- [montamedifar.ashkan@etu.unistra.fr](mailto:montamedifar.ashkan@etu.unistra.fr)
