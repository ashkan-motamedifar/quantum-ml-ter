# Quantum ML for IoT Intrusion Detection

A QCNN reached 69% accuracy on a zero-day attack class (train: normal+DoS, test: unseen injection), where every supervised classical baseline — SVM, random forest, MLPs — stayed below 3%. An unsupervised autoencoder matched the QCNN at 70%. Averaged across 10 seeds, on ToN_IoT `Network_dataset_11`. Master 1 TER at ICube / Université de Strasbourg, supervised by Fabrice Théoleyre.

Preprint: [report/arxiv.pdf](report/arxiv.pdf) (submitting to arXiv:cs.LG) · Thesis: [report/main.pdf](report/main.pdf) · Defense slides: [MOTAMEDIFAR_ASHKAN.pdf](MOTAMEDIFAR_ASHKAN.pdf) · References: [Hur et al. 2022](https://arxiv.org/abs/2108.00661) (QCNN), [Pérez-Salinas et al. 2020](https://arxiv.org/abs/1907.02085) (re-uploading) · Dataset: [ToN_IoT](https://research.unsw.edu.au/projects/toniot-datasets)

## Run

```bash
python3.11 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# place Network_dataset_11.csv from ToN_IoT into data/

python src/preprocessing/preprocess.py

python src/classical/classical_baselines.py
python src/classical/autoencoder.py

python -m src.quantum.qcnn
python -m src.quantum.data_reuploading

bash scripts/run_qcnn_10seed_A.sh
bash scripts/run_qcnn_10seed_B.sh
bash scripts/run_reup_10seed_A.sh
bash scripts/run_reup_10seed_B.sh
bash scripts/summarize_10seed.sh
```
