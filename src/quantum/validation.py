"""
Validation Experiment — Synthetic Dataset
Validates our quantum classifier implementation by reproducing
known results on standard benchmarks from the literature.

Pérez-Salinas et al. (2020) report 90–98% accuracy on synthetic
2D datasets with the single-qubit data re-uploading classifier.
We test on sklearn's make_circles and make_moons.

Usage:
    python -m src.quantum.validation
"""

import pennylane as qml
from pennylane import numpy as np
from sklearn.datasets import make_circles, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, matthews_corrcoef,
    classification_report
)
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOGS = ROOT / 'results' / 'logs'
LOGS.mkdir(parents=True, exist_ok=True)


# ── Minimal single-qubit re-uploading (2 features, binary) ──────────────────

def make_circuit(n_features, n_layers):
    dev = qml.device('default.qubit', wires=1)

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

    return circuit


def train_and_evaluate(X_train, y_train, X_test, y_test, n_layers, n_epochs,
                       dataset_name):
    """Train a single binary re-uploading classifier and evaluate."""
    n_features = X_train.shape[1]
    circuit = make_circuit(n_features, n_layers)
    np.random.seed(42)  # deterministic param init for the validation experiment
    params = np.random.uniform(-0.1, 0.1, (n_layers, 6))
    opt = qml.AdamOptimizer(stepsize=0.05)

    # Binary labels: class 0 → -1, class 1 → +1
    y_tr_binary = np.where(y_train == 1, 1.0, -1.0)

    print(f"\n{'=' * 55}")
    print(f"  VALIDATION: {dataset_name}")
    print(f"  {n_features} features | {n_layers} layers | "
          f"{6 * n_layers} params | {n_epochs} epochs")
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"{'=' * 55}")

    def cost_fn(p):
        preds = np.array([circuit(x, p) for x in X_train])
        return np.mean((y_tr_binary - preds) ** 2)

    for epoch in range(n_epochs):
        params, loss = opt.step_and_cost(cost_fn, params)
        if epoch % 20 == 0 or epoch == n_epochs - 1:
            # Evaluate on test set
            raw = np.array([float(circuit(x, params)) for x in X_test])
            y_pred = (raw > 0).astype(int)
            acc = accuracy_score(y_test, y_pred)
            print(f"  Epoch {epoch:3d} | Loss: {loss:.4f} | "
                  f"Test Acc: {acc:.4f}")

    # Final evaluation
    raw = np.array([float(circuit(x, params)) for x in X_test])
    y_pred = (raw > 0).astype(int)

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)
    cm  = confusion_matrix(y_test, y_pred)

    print(f"\n  Final — Acc: {acc:.4f} | F1: {f1:.4f} | MCC: {mcc:.4f}")
    print(f"  Confusion matrix:\n{cm}")
    print(classification_report(y_test, y_pred, zero_division=0))

    return {
        'dataset': dataset_name,
        'n_layers': n_layers,
        'n_params': 6 * n_layers,
        'n_epochs': n_epochs,
        'accuracy': float(acc),
        'f1': float(f1),
        'mcc': float(mcc),
        'confusion_matrix': cm.tolist(),
    }


if __name__ == '__main__':
    results = []
    scaler = MinMaxScaler(feature_range=(0, np.pi))

    # --- make_circles ---
    X, y = make_circles(n_samples=400, noise=0.1, factor=0.3, random_state=42)
    X = scaler.fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    r = train_and_evaluate(X_tr, y_tr, X_te, y_te,
                           n_layers=4, n_epochs=80,
                           dataset_name="make_circles")
    results.append(r)

    # --- make_moons ---
    X, y = make_moons(n_samples=400, noise=0.1, random_state=42)
    X = scaler.fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    r = train_and_evaluate(X_tr, y_tr, X_te, y_te,
                           n_layers=4, n_epochs=80,
                           dataset_name="make_moons")
    results.append(r)

    # Save
    out = LOGS / 'validation_results.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out}")

    # Summary
    print("\n" + "=" * 55)
    print("  VALIDATION SUMMARY")
    print("=" * 55)
    for r in results:
        print(f"  {r['dataset']:15s} | Acc: {r['accuracy']:.3f} | "
              f"F1: {r['f1']:.3f} | MCC: {r['mcc']:.3f}")
    print("\n  Pérez-Salinas et al. (2020) report 90–98% on similar tasks.")
    print("  If our results are in that range, the implementation is validated.")
