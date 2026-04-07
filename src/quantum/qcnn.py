"""
Quantum Convolutional Neural Network (QCNN)
Based on: Hur et al. (2022)
Implementation: PennyLane

Architecture: 8 qubits → 3 conv-pool stages → 1 qubit → ⟨Z⟩
Parameters: 27 per binary classifier, 81 total (one-vs-all, 3 classes)

Usage:
    python -m src.quantum.qcnn
"""

import pennylane as qml
from pennylane import numpy as np
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
    """Ansatz 6 from Hur et al.: Ry Rz on each qubit, CNOT, Ry Rz on target."""
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
    dev = qml.device('default.qubit', wires=n_qubits)

    @qml.qnode(dev, interface='autograd')
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

    def _init_circuits(self, classes):
        self.classes = classes
        self.circuits = []
        self.params_list = []
        for _ in classes:
            circuit, n_params = make_qcnn_circuit(self.n_qubits)
            # Small init to mitigate barren plateaus
            params = np.random.uniform(-0.1, 0.1, (n_params,))
            self.circuits.append(circuit)
            self.params_list.append(params)

    def _cost_fn(self, params, circuit, X, y_binary):
        predictions = np.array([circuit(x, params) for x in X])
        return np.mean((y_binary - predictions) ** 2)

    def fit(self, X, y, n_epochs=100, verbose=True):
        classes = np.unique(y)
        self._init_circuits(classes)
        opt = qml.AdamOptimizer(stepsize=self.lr)
        history = {'epoch': [], 'loss': []}

        for epoch in range(n_epochs):
            total_loss = 0

            for i, cls in enumerate(self.classes):
                y_binary = np.where(y == cls, 1.0, -1.0)

                # Fix lambda closure: capture i by default arg
                def cost_fn(p, _i=i):
                    return self._cost_fn(
                        p, self.circuits[_i], X, y_binary
                    )

                self.params_list[i], loss = opt.step_and_cost(
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

def load_standard():
    X_train = pd.read_csv(DATA / 'X_train_quantum.csv').values
    y_train = pd.read_csv(DATA / 'y_train_quantum.csv').values.ravel()
    X_test  = pd.read_csv(DATA / 'X_test.csv').values[:200]
    y_test  = pd.read_csv(DATA / 'y_test.csv').values.ravel()[:200]
    print(f"[Standard] Train: {X_train.shape} | Test: {X_test.shape}")
    return X_train, y_train, X_test, y_test


def load_zeroday():
    X_train = pd.read_csv(DATA / 'X_train_zeroday.csv').values
    y_train = pd.read_csv(DATA / 'y_train_zeroday.csv').values.ravel()
    X_test  = pd.read_csv(DATA / 'X_test_zeroday.csv').values
    y_test  = pd.read_csv(DATA / 'y_test_zeroday.csv').values.ravel()
    # Subsample zeroday for quantum simulation
    rng = np.random.default_rng(42)
    idx_tr = rng.choice(len(X_train), size=500, replace=False)
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
    results = {}

    # Standard 3-class
    print("=" * 60)
    print("  QCNN — STANDARD EVALUATION")
    print("=" * 60)
    X_tr, y_tr, X_te, y_te = load_standard()

    model = QCNNClassifier(n_qubits=8, lr=0.01)
    history = model.fit(X_tr, y_tr, n_epochs=30)
    results['standard'] = _evaluate(model, X_te, y_te, "QCNN (standard)")
    results['standard']['history'] = history

    # Zero-day
    print("\n" + "=" * 60)
    print("  QCNN — ZERO-DAY EVALUATION")
    print("=" * 60)
    X_tr_zd, y_tr_zd, X_te_zd, y_te_zd = load_zeroday()

    model_zd = QCNNClassifier(n_qubits=8, lr=0.01)
    history_zd = model_zd.fit(X_tr_zd, y_tr_zd, n_epochs=30)
    results['zeroday'] = _evaluate(model_zd, X_te_zd, y_te_zd, "QCNN (zero-day)")
    results['zeroday']['history'] = history_zd

    # Save
    out = LOGS / 'qcnn_results.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out}")
