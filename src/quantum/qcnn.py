"""
Quantum Convolutional Neural Network (QCNN)
Based on: Hur et al. (2022)
Implementation: PennyLane

Architecture: 8 qubits → 3 conv-pool stages → 1 qubit → ⟨Z⟩
Parameters: 27 per binary classifier, 81 total (one-vs-all, 3 classes)

Usage:
    python -m src.quantum.qcnn
"""

import argparse
import pennylane as qml
from pennylane import numpy as np
import numpy as std_np
import pandas as pd
import json
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, matthews_corrcoef
)

ROOT    = Path(__file__).resolve().parents[2]
DATA    = ROOT / 'data'
LOGS    = ROOT / 'results' / 'logs'
LOGS.mkdir(parents=True, exist_ok=True)


# ── Circuit building blocks ──────────────────────────────────────────────────

def _two_qubit_gate(params, wires):
    """6-parameter Hur-inspired block: RY/RZ on both qubits, CNOT, then RY/RZ on target."""
    qml.RY(params[0], wires=wires[0])
    qml.RZ(params[1], wires=wires[0])
    qml.RY(params[2], wires=wires[1])
    qml.RZ(params[3], wires=wires[1])
    qml.CNOT(wires=wires)
    qml.RY(params[4], wires=wires[1])
    qml.RZ(params[5], wires=wires[1])


def conv_layer(params, wires):
    """Convolutional layer: same 2-qubit gate on all adjacent pairs."""
    n = len(wires)
    for i in range(0, n - 1, 2):
        _two_qubit_gate(params, wires=[wires[i], wires[i + 1]])
    for i in range(1, n - 1, 2):
        _two_qubit_gate(params, wires=[wires[i], wires[i + 1]])


def pool_layer(params, wires):
    """Pooling layer: CRZ + CRY on pairs, keep second qubit."""
    remaining = []
    for i in range(0, len(wires) - 1, 2):
        qml.CRZ(params[0], wires=[wires[i], wires[i + 1]])
        qml.CRY(params[1], wires=[wires[i], wires[i + 1]])
        remaining.append(wires[i + 1])
    return remaining


# ── Full QCNN circuit ────────────────────────────────────────────────────────

def make_qcnn_circuit(n_qubits=8):
    """
    8-qubit QCNN: 3 conv-pool stages + FC layer.
    Total: 3×6 (conv) + 3×2 (pool) + 3 (FC) = 27 parameters.
    """
    dev = qml.device('lightning.qubit', wires=n_qubits)

    @qml.qnode(dev, interface='autograd', diff_method='adjoint')
    def circuit(x, params):
        # Angle encoding
        for i in range(n_qubits):
            qml.RY(x[i], wires=i)

        # Stage 1: 8 → 4
        active = list(range(n_qubits))
        conv_layer(params[0:6], active)
        active = pool_layer(params[6:8], active)

        # Stage 2: 4 → 2
        conv_layer(params[8:14], active)
        active = pool_layer(params[14:16], active)

        # Stage 3: 2 → 1
        conv_layer(params[16:22], active)
        active = pool_layer(params[22:24], active)

        # FC on remaining qubit
        qml.RZ(params[24], wires=active[0])
        qml.RY(params[25], wires=active[0])
        qml.RZ(params[26], wires=active[0])

        return qml.expval(qml.PauliZ(active[0]))

    return circuit, 27


# ── One-vs-All Classifier ────────────────────────────────────────────────────

class QCNNClassifier:
    """3-class QCNN using one-vs-all strategy."""

    def __init__(self, n_qubits=8, lr=0.01):
        self.n_qubits = n_qubits
        self.lr = lr
        self.classes = None
        self.circuits = []
        self.params_list = []

    def _init_circuits(self, classes, seed=42):
        self.classes = classes
        self.circuits = []
        self.params_list = []
        rng = std_np.random.RandomState(seed)
        for _ in classes:
            circuit, n_params = make_qcnn_circuit(self.n_qubits)
            # Small init to mitigate barren plateaus
            params = np.array(rng.uniform(-0.1, 0.1, (n_params,)), requires_grad=True)
            self.circuits.append(circuit)
            self.params_list.append(params)

    def _cost_fn(self, params, circuit, X, y_binary):
        predictions = np.array([circuit(x, params) for x in X])
        return np.mean((y_binary - predictions) ** 2)

    def fit(self, X, y, n_epochs=100, batch_size=None, verbose=True, seed=42):
        classes = np.unique(y)
        self._init_circuits(classes, seed=seed)
        optimizers = [qml.AdamOptimizer(stepsize=self.lr) for _ in classes]
        history = {'epoch': [], 'loss': []}
        rng = np.random.default_rng(0)

        for epoch in range(n_epochs):
            total_loss = 0

            # Mini-batch: use a random subset per epoch for speed
            if batch_size and batch_size < len(X):
                idx = rng.choice(len(X), size=batch_size, replace=False)
                X_b, y_b = X[idx], y[idx]
            else:
                X_b, y_b = X, y

            for i, cls in enumerate(self.classes):
                y_binary = np.where(y_b == cls, 1.0, -1.0)

                # Fix lambda closure: capture i by default arg
                def cost_fn(p, _i=i):
                    return self._cost_fn(
                        p, self.circuits[_i], X_b, y_binary
                    )

                self.params_list[i], loss = optimizers[i].step_and_cost(
                    cost_fn, self.params_list[i]
                )
                total_loss += loss

            avg_loss = total_loss / len(self.classes)
            history['epoch'].append(epoch)
            history['loss'].append(float(avg_loss))

            if verbose and (epoch % 10 == 0 or epoch == n_epochs - 1):
                acc = self.score(X, y)
                print(f"  Epoch {epoch:3d} | Loss: {avg_loss:.4f} | Train Acc: {acc:.4f}")

        return history

    def predict(self, X):
        scores = np.zeros((len(X), len(self.classes)))
        for i, (circuit, params) in enumerate(
            zip(self.circuits, self.params_list)
        ):
            scores[:, i] = np.array([float(circuit(x, params)) for x in X])
        idx = np.argmax(scores, axis=1)
        return np.array([str(self.classes[j]) for j in idx])

    def score(self, X, y):
        y_str = np.array([str(v) for v in y])
        return accuracy_score(y_str, self.predict(X))


# ── Data loading ─────────────────────────────────────────────────────────────

def load_standard(n_samples=500):
    # Always use the same 200 test samples
    X_test  = pd.read_csv(DATA / 'X_test_quantum.csv').values
    y_test  = pd.read_csv(DATA / 'y_test_quantum.csv').values.ravel()

    if n_samples == 500:
        # Use pre-saved subset (same as classical baselines)
        X_train = pd.read_csv(DATA / 'X_train_quantum.csv').values
        y_train = pd.read_csv(DATA / 'y_train_quantum.csv').values.ravel()
    else:
        # Ablation: stratified subsample from full training set
        X_all = pd.read_csv(DATA / 'X_train.csv').values
        y_all = pd.read_csv(DATA / 'y_train.csv').values.ravel()
        from sklearn.model_selection import train_test_split
        _, X_train, _, y_train = train_test_split(
            X_all, y_all, test_size=n_samples, random_state=42, stratify=y_all
        )

    print(f"[Standard] Train: {X_train.shape} | Test: {X_test.shape}")
    return X_train, y_train, X_test, y_test


def load_zeroday(n_samples=500):
    X_train = pd.read_csv(DATA / 'X_train_zeroday.csv').values
    y_train = pd.read_csv(DATA / 'y_train_zeroday.csv').values.ravel()
    X_test  = pd.read_csv(DATA / 'X_test_zeroday.csv').values
    y_test  = pd.read_csv(DATA / 'y_test_zeroday.csv').values.ravel()
    rng = np.random.default_rng(42)
    idx_tr = rng.choice(len(X_train), size=n_samples, replace=False)
    idx_te = rng.choice(len(X_test), size=200, replace=False)
    X_train, y_train = X_train[idx_tr], y_train[idx_tr]
    X_test, y_test   = X_test[idx_te], y_test[idx_te]
    print(f"[Zero-day] Train: {X_train.shape} | Test: {X_test.shape}")
    return X_train, y_train, X_test, y_test


# ── Evaluation ───────────────────────────────────────────────────────────────

def _evaluate(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    # Ensure both are plain string arrays for sklearn
    y_true = np.array([str(v) for v in y_test])
    y_pred = np.array([str(v) for v in y_pred])
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    cm  = confusion_matrix(y_true, y_pred)
    print(f"\n  {name} — Acc: {acc:.4f} | F1: {f1:.4f} | MCC: {mcc:.4f}")
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


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_samples', type=int, default=500,
                        help='Number of training samples (default: 500)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for parameter init (default: 42)')
    args = parser.parse_args()
    N = args.n_samples
    S = args.seed
    BS = 100 if N > 500 else None  # mini-batch for large datasets

    results = {}

    # Standard 3-class
    print("=" * 60)
    print(f"  QCNN — STANDARD  (n_samples={N}, seed={S}, batch={BS})")
    print("=" * 60)
    X_tr, y_tr, X_te, y_te = load_standard(n_samples=N)

    model = QCNNClassifier(n_qubits=8, lr=0.01)
    history = model.fit(X_tr, y_tr, n_epochs=100, batch_size=BS, seed=S)
    results['standard'] = _evaluate(model, X_te, y_te, "QCNN (standard)")
    results['standard']['history'] = history

    # Zero-day
    print("\n" + "=" * 60)
    print(f"  QCNN — ZERO-DAY  (n_samples={N}, seed={S}, batch={BS})")
    print("=" * 60)
    X_tr_zd, y_tr_zd, X_te_zd, y_te_zd = load_zeroday(n_samples=N)

    model_zd = QCNNClassifier(n_qubits=8, lr=0.01)
    history_zd = model_zd.fit(X_tr_zd, y_tr_zd, n_epochs=100, batch_size=BS, seed=S)
    results['zeroday'] = _evaluate(model_zd, X_te_zd, y_te_zd, "QCNN (zero-day)")
    results['zeroday']['history'] = history_zd

    # Save with sample size and seed in filename
    out = LOGS / f'qcnn_{N}s_seed{S}.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out}")
