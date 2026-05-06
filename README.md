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

Classical baselines: SVM, Random Forest, two neural networks (supervised), and an autoencoder anomaly detector (unsupervised). Quantum and autoencoder results averaged over 10 seeds.

---

## Results summary

### Standard 3-class classification

| Model | Accuracy | MCC |
|---|---|---|
| Random Forest | 0.990 | 0.985 |
| SVM (RBF) | 0.970 | 0.955 |
| NN-Medium | 0.960 | 0.940 |
| NN-Small | 0.925 | 0.888 |
| Re-uploading 1q | 0.752 ± 0.039 | 0.662 ± 0.045 |
| Re-uploading 4q | 0.692 ± 0.011 | 0.586 ± 0.016 |
| QCNN (8q) | 0.679 ± 0.025 | 0.559 ± 0.043 |

### Zero-day detection (trained on normal+DoS, tested on injection)

| Model | Accuracy |
|---|---|
| Autoencoder (10 seeds) | **0.705 ± 0.144** |
| QCNN (8q) — 7 seeds (excl. barren plateau) | **0.686 ± 0.068** |
| QCNN (8q) — 10 seeds (all) | 0.490 ± 0.305 |
| Re-uploading 1q | 0.092 ± 0.036 |
| Re-uploading 4q | 0.075 ± 0.023 |
| Best supervised classical (NN-Small) | 0.028 |

**Main finding:** every supervised classical model fails on zero-day (<3%). Two approaches reach ~70% — the autoencoder (70.5%, trained unsupervised on normal traffic only) and the QCNN excluding barren-plateau seeds (68.6%, trained supervised on normal-vs-DoS but transferring to injection). The QCNN reaches this with 56× less training data and lower variance when convergent, but suffers a 30% barren-plateau failure rate (vs. 10% for the autoencoder).

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
│   ├── classical/       SVM, Random Forest, NN, autoencoder
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

# 2. Data: download Network_dataset_11.csv from the ToN_IoT benchmark
#    (https://research.unsw.edu.au/projects/toniot-datasets) and place it at
#    data/Network_dataset_11.csv before running preprocessing.

# 3. Preprocess
python src/preprocessing/preprocess.py

# 4. Classical baselines
python src/classical/classical_baselines.py
python src/classical/autoencoder.py    # 10 seeds → autoencoder_results_10seed.json

# 5. Quantum classifiers (500 samples, 100 epochs, seed 42)
python -m src.quantum.qcnn
python -m src.quantum.data_reuploading

# 6. Multi-seed evaluation (10 seeds, 4 parallel terminals)
bash scripts/run_qcnn_10seed_A.sh    # seeds 0-4
bash scripts/run_qcnn_10seed_B.sh    # seeds 5-9
bash scripts/run_reup_10seed_A.sh    # seeds 0-4
bash scripts/run_reup_10seed_B.sh    # seeds 5-9
bash scripts/summarize_10seed.sh     # aggregate results

# 7. Build report
cd report && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

---

**Ashkan Motamedifar**
