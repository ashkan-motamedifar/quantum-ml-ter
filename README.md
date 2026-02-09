# Quantum Computing for Machine Learning

**TER 2025–2026 — University of Strasbourg (ICube, UMR CNRS 7357)**

Supervisor: Fabrice Théoleyre ([fabrice.theoleyre@cnrs.fr](mailto:fabrice.theoleyre@cnrs.fr))

## Overview

This project explores the application of quantum computing to machine learning, specifically for network traffic classification and zero-day attack detection. We implement and compare classical and quantum classifiers on a real-world IoT network dataset using PennyLane.

## Project Structure

```
quantum-ml-ter/
├── report/                  # LaTeX report (TER memoir)
│   ├── main.tex             # Main document
│   ├── chapters/            # Individual chapters
│   ├── figures/             # Report figures
│   └── references.bib       # Bibliography
├── src/                     # Source code
│   ├── preprocessing/       # Data cleaning, rebalancing
│   ├── classical/           # Classical ML baselines (NN, SVM)
│   ├── quantum/             # Quantum classifiers (PennyLane)
│   └── evaluation/          # Metrics, comparison, zero-day
├── data/                    # Datasets (not tracked by git)
├── results/                 # Experiment outputs
│   ├── figures/             # Generated plots
│   └── logs/                # Training logs
└── notebooks/               # Jupyter notebooks for exploration
```

## Dataset

**Network_dataset_11**: 1,000,000 IoT network flow records with 3 classes:
- Normal traffic (3.5%)
- DoS attacks (84.0%)
- Injection attacks (12.5%)

## Quantum Architectures

- **Data Re-uploading** (Pérez-Salinas et al., 2020)
- **Quantum CNN** (Hur et al., 2022)

## Requirements

```bash
pip install -r requirements.txt
```

## Building the Report

```bash
cd report
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
