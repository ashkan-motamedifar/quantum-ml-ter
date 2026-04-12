"""
Preprocessing pipeline for ToN_IoT Network_dataset_11
TER: Quantum Computing for Machine Learning
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import f_classif

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data'
FIGURES = ROOT / 'results' / 'figures'
FIGURES.mkdir(parents=True, exist_ok=True)

# Load
df = pd.read_csv(DATA / 'Network_dataset_11.csv', encoding='utf-8-sig')
print(f"Shape: {df.shape}")
print(df['type'].value_counts())

# Drop useless columns
useless_columns = [
    'ts', 'src_ip', 'dst_ip', 'missed_bytes', 'label',
    'ssl_version', 'ssl_cipher', 'ssl_resumed', 'ssl_established',
    'ssl_subject', 'ssl_issuer',
    'http_trans_depth', 'http_method', 'http_uri', 'http_referrer',
    'http_version', 'http_request_body_len', 'http_response_body_len',
    'http_status_code', 'http_user_agent', 'http_orig_mime_types',
    'http_resp_mime_types',
    'weird_name', 'weird_addl', 'weird_notice', 'dns_query',
]
df = df.drop(columns=useless_columns)

# Encode categoricals
df['proto'] = (df['proto'] == 'udp').astype(int)
df['service'] = df['service'].map({'-': 0, 'dns': 1, 'http': 2, 'ssl': 3}).fillna(0).astype(int)
conn_map = {s: i for i, s in enumerate(df['conn_state'].unique())}
df['conn_state'] = df['conn_state'].map(conn_map).astype(int)
for col in ['dns_AA', 'dns_RD', 'dns_RA', 'dns_rejected']:
    df[col] = (df[col] == 'T').astype(int)

# ── Step 1: Balance classes by random undersampling to minority size ────────
X_all = df.drop(columns=['type'])
y_all = df['type']
min_size = y_all.value_counts().min()
print(f"Minority class size: {min_size}")

dfs = []
rng = np.random.RandomState(42)
for cls in y_all.unique():
    idx = y_all[y_all == cls].index
    if len(idx) > min_size:
        idx = rng.choice(idx, size=min_size, replace=False)
    dfs.append(pd.concat([X_all.loc[idx], y_all.loc[idx]], axis=1))
df_balanced = pd.concat(dfs, ignore_index=True).sample(frac=1, random_state=42)

X_bal = df_balanced.drop(columns=['type']).reset_index(drop=True)
y_bal = df_balanced['type'].reset_index(drop=True)
print(f"Balanced: {X_bal.shape}\n{y_bal.value_counts()}")

# ── Step 2: Train/test split BEFORE any feature engineering ─────────────────
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_bal, y_bal, test_size=0.2, random_state=42, stratify=y_bal
)
X_train_raw = X_train_raw.reset_index(drop=True)
X_test_raw = X_test_raw.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)
print(f"Train: {X_train_raw.shape}, Test: {X_test_raw.shape}")

# ── Step 3: Remove highly correlated features (fit on TRAIN only) ───────────
upper = X_train_raw.corr().abs().where(
    np.triu(np.ones_like(X_train_raw.corr(), dtype=bool), k=1)
)
high_corr = [col for col in upper.columns if any(upper[col] > 0.95)]
X_train_raw = X_train_raw.drop(columns=high_corr)
X_test_raw = X_test_raw.drop(columns=high_corr)
print(f"Dropped correlated: {high_corr}")

# ── Step 4: Select top 8 features by ANOVA F-score (fit on TRAIN only) ─────
N_QUBITS = 8
f_scores, _ = f_classif(X_train_raw, y_train)
top_features = (
    pd.DataFrame({'feature': X_train_raw.columns, 'f_score': f_scores})
    .sort_values('f_score', ascending=False)
    .head(N_QUBITS)['feature'].tolist()
)
print(f"Top {N_QUBITS} features: {top_features}")

X_train_raw = X_train_raw[top_features]
X_test_raw = X_test_raw[top_features]

# ── Step 5: Normalize to [0, pi] (fit on TRAIN only) ───────────────────────
scaler = MinMaxScaler(feature_range=(0, np.pi))
X_train = pd.DataFrame(scaler.fit_transform(X_train_raw), columns=top_features)
X_test = pd.DataFrame(scaler.transform(X_test_raw), columns=top_features)

# ── Visualizations ──────────────────────────────────────────────────────────
colors = {'normal': '#2ecc71', 'dos': '#e74c3c', 'injection': '#3498db'}

fig, ax = plt.subplots(figsize=(6, 4))
counts = y_bal.value_counts()
bars = ax.bar(counts.index, counts.values, color=[colors[t] for t in counts.index])
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 500,
            f'{val:,}\n({val/len(y_bal)*100:.1f}%)', ha='center', fontsize=9)
ax.set_title('Class Distribution')
ax.set_ylabel('Samples')
plt.tight_layout()
plt.savefig(FIGURES / 'class_distribution.png', dpi=150, bbox_inches='tight')

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(X_train.corr(), mask=np.triu(np.ones((N_QUBITS, N_QUBITS), dtype=bool)),
            cmap='RdBu_r', center=0, square=True, ax=ax)
ax.set_title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig(FIGURES / 'correlation_matrix.png', dpi=150, bbox_inches='tight')

# ── Zero-day split: train on normal+dos, test on unseen injection ───────────
# Use the TRAINING set portion of normal+dos for zero-day training
# Use the TEST set portion of injection for zero-day testing
mask_tr_zd = y_train.isin(['normal', 'dos'])
X_train_zd = X_train[mask_tr_zd]
y_train_zd = y_train[mask_tr_zd].map({'normal': 0, 'dos': 1})

mask_te_zd = y_test == 'injection'
X_test_zd = X_test[mask_te_zd]
y_test_zd = pd.Series(np.ones(X_test_zd.shape[0], dtype=int))

print(f"Zero-day — Train: {X_train_zd.shape}, Test: {X_test_zd.shape}")

# ── Quantum subsample (same subset for classical and quantum) ───────────────
X_train_q = X_train.sample(n=500, random_state=42)
y_train_q = y_train.loc[X_train_q.index]
X_test_q = X_test.sample(n=min(200, len(X_test)), random_state=42)
y_test_q = y_test.loc[X_test_q.index]

# ── Save ────────────────────────────────────────────────────────────────────
X_train.to_csv(DATA / 'X_train.csv', index=False)
X_test.to_csv(DATA / 'X_test.csv', index=False)
y_train.to_csv(DATA / 'y_train.csv', index=False)
y_test.to_csv(DATA / 'y_test.csv', index=False)
X_train_q.to_csv(DATA / 'X_train_quantum.csv', index=False)
y_train_q.to_csv(DATA / 'y_train_quantum.csv', index=False)
X_test_q.to_csv(DATA / 'X_test_quantum.csv', index=False)
y_test_q.to_csv(DATA / 'y_test_quantum.csv', index=False)
X_train_zd.to_csv(DATA / 'X_train_zeroday.csv', index=False)
y_train_zd.to_csv(DATA / 'y_train_zeroday.csv', index=False)
X_test_zd.to_csv(DATA / 'X_test_zeroday.csv', index=False)
y_test_zd.to_csv(DATA / 'y_test_zeroday.csv', index=False)

print(f"\nSaved files:")
print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")
print(f"  X_train_quantum: {X_train_q.shape}, X_test_quantum: {X_test_q.shape}")
print(f"  X_train_zeroday: {X_train_zd.shape}, X_test_zeroday: {X_test_zd.shape}")
print("Done. Data saved to data/")
