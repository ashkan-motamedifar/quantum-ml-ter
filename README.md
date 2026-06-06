# Quantum ML for IoT Intrusion Detection

TER 2025-2026 — University of Strasbourg / ICube (UMR CNRS 7357).
Supervisor: [Fabrice Théoleyre](mailto:fabrice.theoleyre@cnrs.fr).

Comparing classical and quantum classifiers on `Network_dataset_11` from the ToN_IoT benchmark (~1M IoT flows, classes: normal / DoS / injection). The setting of interest is zero-day: train on normal+DoS, test on injection.

Two quantum models implemented in PennyLane:
- QCNN, 8 qubits, 3 conv-pool stages (Hur et al., 2022)
- Data re-uploading, 1q and 4q variants (Pérez-Salinas et al., 2020)

Classical baselines: SVM, Random Forest, two MLPs (NN-Small, NN-Medium), autoencoder. Quantum and autoencoder numbers are averaged over 10 seeds.

## Results

3-class classification:

| Model | Acc | MCC |
|---|---|---|
| Random Forest | 0.990 | 0.985 |
| SVM (RBF) | 0.970 | 0.955 |
| NN-Medium | 0.960 | 0.940 |
| NN-Small | 0.925 | 0.888 |
| Re-uploading 1q | 0.752 ± 0.039 | 0.662 ± 0.045 |
| Re-uploading 4q | 0.692 ± 0.011 | 0.586 ± 0.016 |
| QCNN (8q) | 0.679 ± 0.025 | 0.559 ± 0.043 |

Zero-day (train: normal+DoS, test: injection):

| Model | Acc |
|---|---|
| Autoencoder, 10 seeds | 0.705 ± 0.144 |
| QCNN, 7 convergent seeds | 0.686 ± 0.068 |
| QCNN, all 10 seeds | 0.490 ± 0.305 |
| Re-uploading 1q | 0.092 ± 0.036 |
| Re-uploading 4q | 0.075 ± 0.023 |
| NN-Small (best supervised classical) | 0.028 |

Every supervised classical model collapses on the unseen attack (<3%). Two approaches reach ~70%: the autoencoder (unsupervised, trained on normal traffic) and the QCNN when it doesn't hit a barren plateau (7/10 seeds). The QCNN gets there with ~56× less training data but fails to converge 30% of the time.

## Setup

```bash
python3.11 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Download `Network_dataset_11.csv` from the [ToN_IoT benchmark](https://research.unsw.edu.au/projects/toniot-datasets) into `data/`.

## Run

```bash
python src/preprocessing/preprocess.py

python src/classical/classical_baselines.py
python src/classical/autoencoder.py

python -m src.quantum.qcnn
python -m src.quantum.data_reuploading

# 10-seed runs (split across 4 terminals if you want them parallel)
bash scripts/run_qcnn_10seed_A.sh
bash scripts/run_qcnn_10seed_B.sh
bash scripts/run_reup_10seed_A.sh
bash scripts/run_reup_10seed_B.sh
bash scripts/summarize_10seed.sh
```

Report sources are in `report/`. Build with `pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex`.

Ashkan Motamedifar
