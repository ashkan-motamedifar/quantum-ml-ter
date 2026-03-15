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

# Remove highly correlated features (>0.95)
X = df.drop(columns=['type'])
upper = X.corr().abs().where(np.triu(np.ones_like(X.corr(), dtype=bool), k=1))
high_corr = [col for col in upper.columns if any(upper[col] > 0.95)]
X = X.drop(columns=high_corr)
print(f"Dropped correlated: {high_corr}")

# Select top 8 features by ANOVA F-score
N_QUBITS = 8
f_scores, _ = f_classif(X, df['type'])
top_features = (
    pd.DataFrame({'feature': X.columns, 'f_score': f_scores})
    .sort_values('f_score', ascending=False)
    .head(N_QUBITS)['feature'].tolist()
)
print(f"Top {N_QUBITS} features: {top_features}")

# Balance classes by undersampling to minority size
X = X[top_features]
y = df['type']
min_size = y.value_counts().min()
dfs = []
for cls in y.unique():
    idx = y[y == cls].index
    if len(idx) > min_size:
        idx = idx[:min_size]
    dfs.append(pd.concat([X.loc[idx], y.loc[idx]], axis=1))
df_balanced = pd.concat(dfs, ignore_index=True).sample(frac=1, random_state=42)
X_bal = df_balanced[top_features]
y_bal = df_balanced['type']
print(f"Balanced: {df_balanced.shape}\n{y_bal.value_counts()}")

# Normalize to [0, pi] for angle encoding
scaler = MinMaxScaler(feature_range=(0, np.pi))
X_scaled = pd.DataFrame(scaler.fit_transform(X_bal), columns=top_features)

# Visualizations
colors = {'normal': '#2ecc71', 'dos': '#e74c3c', 'injection': '#3498db'}

fig, ax = plt.subplots(figsize=(6, 4))
counts = y_bal.value_counts()
bars = ax.bar(counts.index, counts.values, color=[colors[t] for t in counts.index])
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 500,
            f'{val:,}\n({val/len(df)*100:.1f}%)', ha='center', fontsize=9)
ax.set_title('Class Distribution')
ax.set_ylabel('Samples')
plt.tight_layout()
plt.savefig(FIGURES / 'class_distribution.png', dpi=150, bbox_inches='tight')

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(X_scaled.corr(), mask=np.triu(np.ones((N_QUBITS, N_QUBITS), dtype=bool)),
            cmap='RdBu_r', center=0, square=True, ax=ax)
ax.set_title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig(FIGURES / 'correlation_matrix.png', dpi=150, bbox_inches='tight')

# Standard train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_bal, test_size=0.2, random_state=42, stratify=y_bal
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# Zero-day split: train on normal+dos, test on unseen injection
mask_tr = y_bal.isin(['normal', 'dos'])
X_train_zd = X_scaled[mask_tr]
y_train_zd = y_bal[mask_tr].map({'normal': 0, 'dos': 1})
X_test_zd = X_scaled[y_bal == 'injection']
y_test_zd = pd.Series(np.ones(X_test_zd.shape[0], dtype=int))
print(f"Zero-day — Train: {X_train_zd.shape}, Test: {X_test_zd.shape}")

# Quantum subsample (simulation constraints)
X_train_q = X_train.sample(n=500, random_state=42)
y_train_q = y_train.loc[X_train_q.index]
X_test_q = X_test.sample(n=min(200, len(X_test)), random_state=42)
y_test_q = y_test.loc[X_test_q.index]

# Save
X_train.to_csv(DATA / 'X_train.csv', index=False)
X_test.to_csv(DATA / 'X_test.csv', index=False)
y_train.to_csv(DATA / 'y_train.csv', index=False)
y_test.to_csv(DATA / 'y_test.csv', index=False)
X_train_q.to_csv(DATA / 'X_train_quantum.csv', index=False)
y_train_q.to_csv(DATA / 'y_train_quantum.csv', index=False)
X_train_zd.to_csv(DATA / 'X_train_zeroday.csv', index=False)
y_train_zd.to_csv(DATA / 'y_train_zeroday.csv', index=False)
X_test_zd.to_csv(DATA / 'X_test_zeroday.csv', index=False)
y_test_zd.to_csv(DATA / 'y_test_zeroday.csv', index=False)

print("Done. Data saved to data/")
