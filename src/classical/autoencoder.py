"""
Autoencoder Baseline for Zero-Day Detection
TER: Quantum Computing for Machine Learning — IoT Intrusion Detection

Unsupervised anomaly detection:
    - Train autoencoder on NORMAL traffic only (no labels needed at inference time)
    - Compute reconstruction error threshold from training distribution
    - At test time, flag samples with error > threshold as anomalies
    - Evaluate on unseen injection attacks

This is the standard "anomaly-detection" approach to zero-day intrusion
detection, complementing the supervised classical baselines which all fail
below 3% on this task.
"""

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data'
LOGS = ROOT / 'results' / 'logs'
LOGS.mkdir(parents=True, exist_ok=True)


# ── Model ────────────────────────────────────────────────────────────────────

class Autoencoder(nn.Module):
    """Symmetric autoencoder: 8 → 6 → 3 → 6 → 8 (tanh activations).

    Design choices (after comparing against a ReLU variant on 5 seeds):
      - tanh avoids dead-unit failures observed with ReLU on ~40% of seeds
      - bottleneck 3 (< 8/2) forces a genuine compression of normal traffic
      - Xavier init keeps activations in the linear regime of tanh at start
    """

    def __init__(self, n_features=8, hidden=6, latent=3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden), nn.Tanh(),
            nn.Linear(hidden, latent),    nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, hidden), nn.Tanh(),
            nn.Linear(hidden, n_features),
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.decoder(self.encoder(x))


# ── Helpers ──────────────────────────────────────────────────────────────────

def reconstruction_error(model, X, batch=1024):
    """Per-sample MSE between x and model(x)."""
    model.eval()
    errs = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = X[i:i+batch]
            xh = model(xb)
            errs.append(((xh - xb) ** 2).mean(dim=1).cpu().numpy())
    return np.concatenate(errs)


def run(seed=42, percentile=95):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load zero-day data
    X_train = pd.read_csv(DATA / 'X_train_zeroday.csv').values.astype(np.float32)
    y_train = pd.read_csv(DATA / 'y_train_zeroday.csv').values.ravel()
    X_test  = pd.read_csv(DATA / 'X_test_zeroday.csv').values.astype(np.float32)
    y_test  = pd.read_csv(DATA / 'y_test_zeroday.csv').values.ravel()

    # Keep ONLY normal for training (y_train == 0)
    X_normal = X_train[y_train == 0]
    X_dos    = X_train[y_train == 1]
    print(f"[Autoencoder] Normal train: {X_normal.shape} | DoS (holdout): {X_dos.shape}")
    print(f"              Injection test: {X_test.shape}")

    # Split normal into train / val (90/10)
    rng  = np.random.default_rng(seed)
    idx  = rng.permutation(len(X_normal))
    cut  = int(0.9 * len(idx))
    Xtr  = torch.tensor(X_normal[idx[:cut]])
    Xval = torch.tensor(X_normal[idx[cut:]])
    Xdos = torch.tensor(X_dos)
    Xte  = torch.tensor(X_test)

    # Train
    model = Autoencoder(n_features=Xtr.shape[1])
    opt   = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)
    loss_fn = nn.MSELoss()

    n_epochs = 100
    batch    = 128
    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        total = 0.0
        for i in range(0, len(Xtr), batch):
            xb = Xtr[perm[i:i+batch]]
            opt.zero_grad()
            loss = loss_fn(model(xb), xb)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        sched.step()
        if (epoch + 1) % 20 == 0:
            val_err = reconstruction_error(model, Xval).mean()
            print(f"  epoch {epoch+1:3d} | train MSE {total/len(Xtr):.5f} | val MSE {val_err:.5f}")

    # Threshold: 95th percentile of reconstruction errors on TRAINING normals
    train_errs = reconstruction_error(model, Xtr)
    threshold  = float(np.percentile(train_errs, percentile))
    print(f"\n  Threshold ({percentile}th pct of train errors): {threshold:.5f}")

    # Evaluate
    # 1) False positive rate on held-out normal validation
    val_errs = reconstruction_error(model, Xval)
    fpr_normal = float((val_errs > threshold).mean())

    # 2) Detection rate on DoS (seen-attack sanity check)
    dos_errs = reconstruction_error(model, Xdos)
    det_dos  = float((dos_errs > threshold).mean())

    # 3) Detection rate on INJECTION (unseen, zero-day)
    test_errs = reconstruction_error(model, Xte)
    det_injection = float((test_errs > threshold).mean())

    # Test set is all-injection (label=1); "accuracy" == detection rate
    y_pred = (test_errs > threshold).astype(int)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, zero_division=0)

    print("\n" + "=" * 60)
    print("  AUTOENCODER — ZERO-DAY RESULTS")
    print("=" * 60)
    print(f"  FPR on held-out normals   : {fpr_normal:.4f}  (expected ~{(100-percentile)/100:.2f})")
    print(f"  Detection rate on DoS     : {det_dos:.4f}  (sanity check, seen class)")
    print(f"  Detection rate on INJECT. : {det_injection:.4f}  (zero-day)")
    print(f"  Test accuracy             : {acc:.4f}")
    print(f"  F1 score                  : {f1:.4f}")

    return {
        'model': 'Autoencoder',
        'seed': seed,
        'percentile_threshold': percentile,
        'threshold': threshold,
        'fpr_normal': fpr_normal,
        'detection_dos': det_dos,
        'detection_injection': det_injection,
        'accuracy': float(acc),
        'f1': float(f1),
        'architecture': '8-6-3-6-8',
        'n_epochs': n_epochs,
    }


def run_multiseed(seeds=(0, 1, 2, 3, 4, 5, 6, 7, 8, 42), percentile=95):
    results = []
    for s in seeds:
        print(f"\n--- seed {s} ---")
        results.append(run(seed=s, percentile=percentile))

    accs = np.array([r['accuracy'] for r in results])
    dets = np.array([r['detection_injection'] for r in results])
    fprs = np.array([r['fpr_normal'] for r in results])

    print("\n" + "=" * 60)
    print(f"  MULTI-SEED SUMMARY  (seeds: {list(seeds)})")
    print("=" * 60)
    print(f"  Injection detection : {dets.mean():.4f} ± {dets.std():.4f}")
    print(f"  FPR (false alarms)  : {fprs.mean():.4f} ± {fprs.std():.4f}")
    print(f"  Accuracy            : {accs.mean():.4f} ± {accs.std():.4f}")

    summary = {
        'model': 'Autoencoder',
        'seeds': list(seeds),
        'percentile_threshold': percentile,
        'detection_injection_mean': float(dets.mean()),
        'detection_injection_std':  float(dets.std()),
        'fpr_mean': float(fprs.mean()),
        'fpr_std':  float(fprs.std()),
        'accuracy_mean': float(accs.mean()),
        'accuracy_std':  float(accs.std()),
        'per_seed': results,
    }
    return summary


if __name__ == '__main__':
    # Default: 10 seeds matching the report — (0,1,2,3,4,5,6,7,8,42)
    summary = run_multiseed()
    out_path = LOGS / 'autoencoder_results_10seed.json'
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved → {out_path}")
