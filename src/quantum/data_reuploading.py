"""
Data Re-uploading Quantum Classifier
Based on: Pérez-Salinas et al. (2020)
Implementation: PennyLane

Two variants:
    - Single-qubit (1 qubit, 6 layers, 36 params/classifier)
    - Multi-qubit  (4 qubits, 3 layers, 72 params/classifier)

Usage:
    python -m src.quantum.data_reuploading
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


# ── Single-qubit circuit ─────────────────────────────────────────────────────

def make_reuploading_circuit(n_features, n_layers):
    """Single-qubit data re-uploading. 6 params per layer."""
    dev = qml.device('default.qubit', wires=1)
    n_params = 6 * n_layers

    @qml.qnode(dev, interface='autograd')
    def circuit(x, params):
        for l in range(n_layers):
            biases  = params[l, :3]
            weights = params[l, 3:]

            f1 = x[(3 * l + 0) % n_features]
            f2 = x[(3 * l + 1) % n_features]
            f3 = x[(3 * l + 2) % n_features]

            qml.RZ(biases[0] + weights[0] * f1, wires=0)
            qml.RY(biases[1] + weights[1] * f2, wires=0)
            qml.RZ(biases[2] + weights[2] * f3, wires=0)

        return qml.expval(qml.PauliZ(0))

    return circuit, n_params


# ── Multi-qubit circuit ──────────────────────────────────────────────────────

def make_multiqubit_reuploading_circuit(n_features, n_qubits, n_layers):
    """Multi-qubit re-uploading with CZ entanglement. Measures ALL qubits."""
    dev = qml.device('default.qubit', wires=n_qubits)
    n_params = 6 * n_qubits * n_layers

    @qml.qnode(dev, interface='autograd')
    def circuit(x, params):
        for l in range(n_layers):
            for q in range(n_qubits):
                biases  = params[l, q, :3]
                weights = params[l, q, 3:]

                f1 = x[(3 * (l * n_qubits + q) + 0) % n_features]
                f2 = x[(3 * (l * n_qubits + q) + 1) % n_features]
                f3 = x[(3 * (l * n_qubits + q) + 2) % n_features]

                qml.RZ(biases[0] + weights[0] * f1, wires=q)
                qml.RY(biases[1] + weights[1] * f2, wires=q)
                qml.RZ(biases[2] + weights[2] * f3, wires=q)

            # CZ entanglement (nearest-neighbor)
            for q in range(n_qubits - 1):
                qml.CZ(wires=[q, q + 1])

        # Average ⟨Z⟩ over all qubits (build Hamiltonian, then measure once)
        coeffs = [1.0 / n_qubits] * n_qubits
        obs    = [qml.PauliZ(q) for q in range(n_qubits)]
        H      = qml.Hamiltonian(coeffs, obs)
        return qml.expval(H)

    return circuit, n_params


# ── One-vs-All Classifier ────────────────────────────────────────────────────

class ReuploadingClassifier:
    """3-class classifier using one-vs-all re-uploading circuits."""

    def __init__(self, n_features=8, n_layers=6, n_qubits=1, lr=0.01):
        self.n_features = n_features
        self.n_layers = n_layers
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
            if self.n_qubits == 1:
                circuit, _ = make_reuploading_circuit(
                    self.n_features, self.n_layers
                )
                # Small init for barren plateaus
                params = np.random.uniform(
                    -0.1, 0.1, (self.n_layers, 6)
                )
            else:
                circuit, _ = make_multiqubit_reuploading_circuit(
                    self.n_features, self.n_qubits, self.n_layers
                )
                params = np.random.uniform(
                    -0.1, 0.1, (self.n_layers, self.n_qubits, 6)
                )

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

                # Fix lambda closure
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
    rng = np.random.default_rng(42)
    idx_tr = rng.choice(len(X_train), size=500, replace=False)
    idx_te = rng.choice(len(X_test), size=200, replace=False)
    X_train, y_train = X_train[idx_tr], y_train[idx_tr]
    X_test, y_test   = X_test[idx_te], y_test[idx_te]
    print(f"[Zero-day] Train: {X_train.shape} | Test: {X_test.shape}")
    return X_train, y_train, X_test, y_test


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

    X_tr, y_tr, X_te, y_te = load_standard()
    n_feat = X_tr.shape[1]

    # --- Single-qubit (6 layers) ---
    print("=" * 60)
    print("  SINGLE-QUBIT RE-UPLOADING — STANDARD")
    print("=" * 60)
    model_1q = ReuploadingClassifier(
        n_features=n_feat, n_layers=6, n_qubits=1, lr=0.01
    )
    h1 = model_1q.fit(X_tr, y_tr, n_epochs=100)
    results['single_qubit_standard'] = _evaluate(model_1q, X_te, y_te, "Re-uploading 1q (standard)")
    results['single_qubit_standard']['history'] = h1

    # --- Multi-qubit (4 qubits, 3 layers) ---
    print("\n" + "=" * 60)
    print("  4-QUBIT RE-UPLOADING — STANDARD")
    print("=" * 60)
    model_4q = ReuploadingClassifier(
        n_features=n_feat, n_layers=3, n_qubits=4, lr=0.01
    )
    h4 = model_4q.fit(X_tr, y_tr, n_epochs=100)
    results['multi_qubit_standard'] = _evaluate(model_4q, X_te, y_te, "Re-uploading 4q (standard)")
    results['multi_qubit_standard']['history'] = h4

    # --- Zero-day ---
    X_tr_zd, y_tr_zd, X_te_zd, y_te_zd = load_zeroday()

    print("\n" + "=" * 60)
    print("  SINGLE-QUBIT RE-UPLOADING — ZERO-DAY")
    print("=" * 60)
    model_1q_zd = ReuploadingClassifier(
        n_features=n_feat, n_layers=6, n_qubits=1, lr=0.01
    )
    h1z = model_1q_zd.fit(X_tr_zd, y_tr_zd, n_epochs=100)
    results['single_qubit_zeroday'] = _evaluate(model_1q_zd, X_te_zd, y_te_zd, "Re-uploading 1q (zero-day)")
    results['single_qubit_zeroday']['history'] = h1z

    print("\n" + "=" * 60)
    print("  4-QUBIT RE-UPLOADING — ZERO-DAY")
    print("=" * 60)
    model_4q_zd = ReuploadingClassifier(
        n_features=n_feat, n_layers=3, n_qubits=4, lr=0.01
    )
    h4z = model_4q_zd.fit(X_tr_zd, y_tr_zd, n_epochs=100)
    results['multi_qubit_zeroday'] = _evaluate(model_4q_zd, X_te_zd, y_te_zd, "Re-uploading 4q (zero-day)")
    results['multi_qubit_zeroday']['history'] = h4z

    # Save
    out = LOGS / 'reuploading_results.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out}")
