"""
Classical Baselines for Quantum ML Comparison
TER: Quantum Computing for Machine Learning — IoT Intrusion Detection

Models:
    - SVM (RBF kernel, grid-searched)
    - Random Forest
    - Neural Network Small  (~207 params)
    - Neural Network Medium (~307 params)

Two evaluation modes:
    - Standard   : 3-class (normal / dos / injection)
    - Zero-day   : train on normal+dos, test on unseen injection
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, matthews_corrcoef
)
from sklearn.preprocessing import LabelEncoder

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parents[2]
DATA    = ROOT / 'data'
RESULTS = ROOT / 'results'
LOGS    = RESULTS / 'logs'
LOGS.mkdir(parents=True, exist_ok=True)


# ── Data loading ─────────────────────────────────────────────────────────────

def load_standard():
    """Load standard 3-class train/test split (quantum subsample)."""
    X_train = pd.read_csv(DATA / 'X_train_quantum.csv').values
    y_train = pd.read_csv(DATA / 'y_train_quantum.csv').values.ravel()
    X_test  = pd.read_csv(DATA / 'X_test_quantum.csv').values
    y_test  = pd.read_csv(DATA / 'y_test_quantum.csv').values.ravel()

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc  = le.transform(y_test)

    print(f"[Standard] Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"           Classes: {list(le.classes_)}\n")
    return X_train, y_train_enc, X_test, y_test_enc, le


def load_zeroday():
    """
    Load zero-day split.
    Train: normal + dos (binary labels)
    Test : injection only (all labeled as attack=1, unseen during training)
    """
    X_train = pd.read_csv(DATA / 'X_train_zeroday.csv').values
    y_train = pd.read_csv(DATA / 'y_train_zeroday.csv').values.ravel()
    X_test  = pd.read_csv(DATA / 'X_test_zeroday.csv').values
    y_test  = pd.read_csv(DATA / 'y_test_zeroday.csv').values.ravel()

    print(f"[Zero-day] Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"           Train classes: {np.unique(y_train)}")
    print(f"           Test  classes: {np.unique(y_test)} (injection — UNSEEN)\n")
    return X_train, y_train, X_test, y_test


# ── Models ───────────────────────────────────────────────────────────────────

def train_svm(X_train, y_train, X_test, y_test, name="SVM"):
    """SVM with RBF kernel, hyperparameters tuned by 5-fold cross-validation."""
    print(f"=== {name} (RBF Kernel) ===")

    param_grid = {
        'C':     [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.01, 0.1],
    }
    grid = GridSearchCV(
        SVC(kernel='rbf', random_state=42, probability=True),
        param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=0
    )
    grid.fit(X_train, y_train)

    print(f"  Best params : {grid.best_params_}")
    print(f"  Best CV acc : {grid.best_score_:.4f}")

    y_pred = grid.predict(X_test)
    return _report(y_test, y_pred, name), grid.best_estimator_


def train_random_forest(X_train, y_train, X_test, y_test, name="Random Forest"):
    """Random Forest — strong classical baseline, important for comparison."""
    print(f"=== {name} ===")

    rf = RandomForestClassifier(
        n_estimators=100, max_depth=None,
        random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    return _report(y_test, y_pred, name), rf


def train_nn(X_train, y_train, X_test, y_test, hidden_layers, name="NN"):
    """Classical MLP — architectures chosen to match quantum parameter counts."""
    n_classes = len(np.unique(y_train))
    layers    = [X_train.shape[1]] + list(hidden_layers) + [n_classes]
    n_params  = sum(layers[i] * layers[i+1] + layers[i+1] for i in range(len(layers)-1))

    print(f"=== {name} ===")
    print(f"  Architecture : {' → '.join(map(str, layers))}")
    print(f"  Parameters   : {n_params}")

    nn = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation='relu',
        solver='adam',
        learning_rate_init=0.001,
        max_iter=200,
        random_state=42,
        verbose=False,
    )
    nn.fit(X_train, y_train)

    y_pred = nn.predict(X_test)
    return _report(y_test, y_pred, name), nn


# ── Evaluation helper ─────────────────────────────────────────────────────────

def _report(y_true, y_pred, name):
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    cm  = confusion_matrix(y_true, y_pred)

    print(f"  Test accuracy : {acc:.4f}")
    print(f"  Weighted F1   : {f1:.4f}")
    print(f"  MCC           : {mcc:.4f}")
    print(f"  Confusion matrix:\n{cm}")
    print(classification_report(y_true, y_pred, zero_division=0))

    report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)

    return {
        'model': name,
        'accuracy': float(acc),
        'f1': float(f1),
        'mcc': float(mcc),
        'confusion_matrix': cm.tolist(),
        'per_class': {
            str(k): {
                'precision': float(v['precision']),
                'recall': float(v['recall']),
                'f1': float(v['f1-score']),
                'support': int(v['support']),
            }
            for k, v in report.items()
            if k not in ('accuracy', 'macro avg', 'weighted avg')
        },
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def run_standard():
    """Run all models on the standard 3-class split."""
    print("\n" + "=" * 60)
    print("  STANDARD EVALUATION  (3-class: normal / dos / injection)")
    print("=" * 60 + "\n")

    X_tr, y_tr, X_te, y_te, le = load_standard()

    results = []
    metrics, _ = train_svm(X_tr, y_tr, X_te, y_te, name="SVM")
    results.append(metrics)

    metrics, _ = train_random_forest(X_tr, y_tr, X_te, y_te)
    results.append(metrics)

    metrics, _ = train_nn(
        X_tr, y_tr, X_te, y_te,
        hidden_layers=(16, 8),
        name="NN-Medium (~307 params)"
    )
    results.append(metrics)

    metrics, _ = train_nn(
        X_tr, y_tr, X_te, y_te,
        hidden_layers=(12, 6),
        name="NN-Small (~207 params)"
    )
    results.append(metrics)

    return results


def run_zeroday():
    """
    Run all models on the zero-day split.
    Train on normal+dos, evaluate on injection (unseen class).
    This simulates real-world zero-day detection.
    """
    print("\n" + "=" * 60)
    print("  ZERO-DAY EVALUATION  (train: normal+dos | test: injection)")
    print("=" * 60 + "\n")

    X_tr, y_tr, X_te, y_te = load_zeroday()

    results = []
    metrics, _ = train_svm(X_tr, y_tr, X_te, y_te, name="SVM (zero-day)")
    results.append(metrics)

    metrics, _ = train_random_forest(X_tr, y_tr, X_te, y_te, name="Random Forest (zero-day)")
    results.append(metrics)

    metrics, _ = train_nn(
        X_tr, y_tr, X_te, y_te,
        hidden_layers=(16, 8),
        name="NN-Medium (zero-day)"
    )
    results.append(metrics)

    metrics, _ = train_nn(
        X_tr, y_tr, X_te, y_te,
        hidden_layers=(12, 6),
        name="NN-Small (zero-day)"
    )
    results.append(metrics)

    return results


def print_summary(standard_results, zeroday_results):
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)

    print("\n[Standard — 3-class]")
    print(f"  {'Model':<45} {'Accuracy':>10} {'F1':>10}")
    print("  " + "-" * 65)
    for r in standard_results:
        print(f"  {r['model']:<45} {r['accuracy']:>10.4f} {r['f1']:>10.4f}")

    print("\n[Zero-day — injection unseen]")
    print(f"  {'Model':<45} {'Accuracy':>10} {'F1':>10}")
    print("  " + "-" * 65)
    for r in zeroday_results:
        print(f"  {r['model']:<45} {r['accuracy']:>10.4f} {r['f1']:>10.4f}")


if __name__ == '__main__':
    std_res = run_standard()
    zd_res  = run_zeroday()
    print_summary(std_res, zd_res)

    # Save results for later comparison with quantum models
    all_results = {'standard': std_res, 'zeroday': zd_res}
    out_path = LOGS / 'classical_results.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved → {out_path}")
