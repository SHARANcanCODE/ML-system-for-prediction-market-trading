import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (roc_auc_score, accuracy_score, classification_report,
                             precision_recall_curve, average_precision_score)
from pathlib import Path
import json
import time
import sys

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / 'data'

WINDOW = 48
HORIZON = 12
BATCH_SIZE = 256
EPOCHS = 20
PATIENCE = 5
LR = 1e-3

if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
elif torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

print(f"Device: {DEVICE}")
print(f"PyTorch: {torch.__version__}")
sys.stdout.flush()

print("\n=== 1. Loading data ===")
sys.stdout.flush()

prices = pd.read_parquet(DATA / 'processed/prices.parquet')
print(f"Prices: {prices.shape}, Tokens: {prices['token_id'].nunique()}")
sys.stdout.flush()

MIN_SEQ_LEN = WINDOW + HORIZON + 10
N_FEATURES = 4

t0 = time.time()
sequences_X, sequences_y = [], []
n_skipped = 0

grouped = prices.groupby('token_id')['price']
n_tokens = len(grouped)
print(f"Processing {n_tokens} tokens...")
sys.stdout.flush()

for idx, (token, price_series) in enumerate(grouped):
    p = price_series.values.astype(np.float32)
    if len(p) < MIN_SEQ_LEN:
        n_skipped += 1
        continue

    ret = np.empty_like(p); ret[0] = 0; ret[1:] = np.diff(p)
    vol = pd.Series(ret).rolling(12, min_periods=1).std().values.astype(np.float32)
    ma12 = pd.Series(p).rolling(12, min_periods=1).mean().values.astype(np.float32)
    momentum = p - ma12
    features = np.column_stack([p, ret, vol, momentum])

    T = len(features)
    n_seq = T - WINDOW - HORIZON
    if n_seq <= 0:
        n_skipped += 1
        continue

    strides = features.strides
    X_windows = np.lib.stride_tricks.as_strided(
        features,
        shape=(n_seq, WINDOW, N_FEATURES),
        strides=(strides[0], strides[0], strides[1])
    ).copy()

    idx_starts = np.arange(n_seq)
    current_p = p[idx_starts + WINDOW - 1]
    future_p = p[idx_starts + WINDOW + HORIZON - 1]
    labels = (future_p > current_p).astype(np.float32)

    valid = ~np.isnan(X_windows.reshape(n_seq, -1)).any(axis=1)
    if valid.sum() > 0:
        sequences_X.append(X_windows[valid])
        sequences_y.append(labels[valid])

    if (idx + 1) % 200 == 0:
        total = sum(len(s) for s in sequences_X)
        elapsed = time.time() - t0
        print(f"  {idx+1}/{n_tokens} tokens | {total:,} sequences | {elapsed:.0f}s")
        sys.stdout.flush()

print("Concatenating...")
sys.stdout.flush()
X = np.concatenate(sequences_X, axis=0)
y = np.concatenate(sequences_y, axis=0)
del sequences_X, sequences_y

elapsed = time.time() - t0
print(f"\nDataset: X={X.shape}, y={y.shape} ({elapsed:.1f}s)")
print(f"Tokens: {n_tokens - n_skipped}/{n_tokens} used")
print(f"Class balance: UP={y.mean():.1%}, DOWN={1-y.mean():.1%}")
sys.stdout.flush()

print("\n=== 2. Split + Normalize ===")
n = len(X)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

X_train, y_train = X[:train_end], y[:train_end]
X_val, y_val = X[train_end:val_end], y[train_end:val_end]
X_test, y_test = X[val_end:], y[val_end:]

print(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
print(f"Train UP: {y_train.mean():.1%} | Test UP: {y_test.mean():.1%}")

train_mean = X_train.reshape(-1, N_FEATURES).mean(axis=0)
train_std = X_train.reshape(-1, N_FEATURES).std(axis=0) + 1e-8
X_train_norm = (X_train - train_mean) / train_std
X_val_norm = (X_val - train_mean) / train_std
X_test_norm = (X_test - train_mean) / train_std
del X
print("Normalized ✓")
sys.stdout.flush()

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.75):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        p_t = torch.where(targets == 1, p, 1 - p)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        focal_weight = (1 - p_t) ** self.gamma
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        return (alpha_t * focal_weight * bce).mean()

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=1), nn.BatchNorm1d(channels), nn.ReLU(),
            nn.Conv1d(channels, channels, 3, padding=1), nn.BatchNorm1d(channels))
    def forward(self, x):
        return F.relu(self.block(x) + x)

class ResCNN(nn.Module):
    def __init__(self, in_channels=4, dropout=0.3):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(in_channels, 64, 7, padding=3), nn.BatchNorm1d(64), nn.ReLU())
        self.res1 = ResBlock(64); self.pool1 = nn.MaxPool1d(2)
        self.res2 = ResBlock(64); self.pool2 = nn.MaxPool1d(2)
        self.res3 = ResBlock(64)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(64, 1))
    def forward(self, x):
        x = self.stem(x)
        x = self.pool1(self.res1(x)); x = self.pool2(self.res2(x)); x = self.res3(x)
        return self.classifier(self.gap(x).squeeze(-1)).squeeze(-1)

class PriceGRU(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.classifier = nn.Sequential(nn.Linear(hidden_size, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1))
    def forward(self, x):
        _, h_n = self.gru(x)
        return self.classifier(h_n[-1]).squeeze(-1)

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, channels_first=False):
        self.X = torch.tensor(X, dtype=torch.float32)
        if channels_first:
            self.X = self.X.transpose(1, 2)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    for X_b, y_b in loader:
        X_b = X_b.to(DEVICE)
        logits = model(X_b)
        all_preds.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(y_b.numpy())
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    auc = roc_auc_score(labels, preds)
    return auc, preds, labels

def train_one(model, train_loader, val_loader, test_loader, criterion, name):
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    best_auc, best_state, no_improve = 0, None, 0
    t0 = time.time()

    for epoch in range(EPOCHS):
        model.train()
        total_loss, n = 0, 0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(y_b); n += len(y_b)

        val_auc, _, _ = evaluate(model, val_loader)
        scheduler.step(total_loss / n)

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= PATIENCE:
            print(f"  [{name}] Early stop ep {epoch+1}, best val AUC={best_auc:.4f}")
            sys.stdout.flush()
            break

    elapsed = time.time() - t0
    model.load_state_dict(best_state)
    test_auc, preds, labels = evaluate(model, test_loader)

    pred_05 = (preds > 0.5).astype(int)
    report = classification_report(labels, pred_05, target_names=['DOWN', 'UP'], output_dict=True)

    prec, rec, thrs = precision_recall_curve(labels, preds)
    f1s = 2 * prec * rec / (prec + rec + 1e-8)
    best_idx = np.argmax(f1s)
    best_thr = thrs[best_idx] if best_idx < len(thrs) else 0.5
    pred_opt = (preds > best_thr).astype(int)
    report_opt = classification_report(labels, pred_opt, target_names=['DOWN', 'UP'], output_dict=True)

    print(f"  [{name}] AUC={test_auc:.4f} | Recall UP@0.5={report['UP']['recall']:.3f} | "
          f"thr={best_thr:.3f}→Recall={report_opt['UP']['recall']:.3f} F1={report_opt['UP']['f1-score']:.3f} | {elapsed:.0f}s")
    sys.stdout.flush()

    return {
        'name': name, 'test_auc': test_auc,
        'ap': average_precision_score(labels, preds),
        'recall_up_05': report['UP']['recall'], 'precision_up_05': report['UP']['precision'],
        'f1_up_05': report['UP']['f1-score'], 'recall_down_05': report['DOWN']['recall'],
        'accuracy_05': report['accuracy'], 'best_threshold': best_thr,
        'recall_up_opt': report_opt['UP']['recall'], 'precision_up_opt': report_opt['UP']['precision'],
        'f1_up_opt': report_opt['UP']['f1-score'], 'accuracy_opt': report_opt['accuracy'],
        'train_time': elapsed,
    }

print("\n=== 3. Running DL experiments ===")
sys.stdout.flush()

pos_weight = torch.tensor([(1 - y_train.mean()) / y_train.mean()]).to(DEVICE)
print(f"pos_weight = {pos_weight.item():.3f}")

loss_configs = [
    ('BCE', nn.BCEWithLogitsLoss()),
    ('W-BCE', nn.BCEWithLogitsLoss(pos_weight=pos_weight)),
    ('Focal_g1', FocalLoss(gamma=1.0, alpha=0.75)),
    ('Focal_g2', FocalLoss(gamma=2.0, alpha=0.75)),
    ('Focal_g3', FocalLoss(gamma=3.0, alpha=0.75)),
]

arch_configs = [
    ('ResCNN', ResCNN, True),
    ('GRU', PriceGRU, False),
]

results = []

for arch_name, ModelClass, ch_first in arch_configs:
    print(f"\n--- {arch_name} ---")
    sys.stdout.flush()

    train_ds = TimeSeriesDataset(X_train_norm, y_train, channels_first=ch_first)
    val_ds = TimeSeriesDataset(X_val_norm, y_val, channels_first=ch_first)
    test_ds = TimeSeriesDataset(X_test_norm, y_test, channels_first=ch_first)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE * 2)

    for loss_name, criterion in loss_configs:
        torch.manual_seed(SEED)
        model = ModelClass()
        result = train_one(model, train_loader, val_loader, test_loader, criterion, f"{arch_name}_{loss_name}")
        results.append(result)

print("\n=== 4. LightGBM experiments ===")
sys.stdout.flush()

import lightgbm as lgb
import joblib

fm = pd.read_parquet(DATA / 'features/feature_matrix_v3.parquet')
with open(DATA / 'features/feature_meta_v3.json') as f:
    meta = json.load(f)
feature_names = meta['features']

X_tab = fm[feature_names].values
y_tab = fm['y'].values

split_tr = int(len(X_tab) * 0.7)
split_vl = int(len(X_tab) * 0.85)
X_tr, y_tr = X_tab[:split_tr], y_tab[:split_tr]
X_vl, y_vl = X_tab[split_tr:split_vl], y_tab[split_tr:split_vl]
X_te, y_te = X_tab[split_vl:], y_tab[split_vl:]

lgb_model = joblib.load(DATA / 'models/lgb_v3.joblib')
base_params = lgb_model.get_params()
base_params['verbose'] = -1; base_params['n_jobs'] = -1

lgb_results = []
for name, extra in [('LGB_baseline', {}), ('LGB_is_unbalance', {'is_unbalance': True}),
                     ('LGB_scale_pos_weight', {'scale_pos_weight': (1 - y_tr.mean()) / y_tr.mean()})]:
    params = {**base_params, **extra}
    m = lgb.LGBMClassifier(**params)
    m.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)],
          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    preds = m.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, preds)
    pred_05 = (preds > 0.5).astype(int)
    rep = classification_report(y_te, pred_05, target_names=['DOWN', 'UP'], output_dict=True)
    prec, rec, thrs = precision_recall_curve(y_te, preds)
    f1s = 2 * prec * rec / (prec + rec + 1e-8)
    bi = np.argmax(f1s)
    bt = thrs[bi] if bi < len(thrs) else 0.5
    pred_opt = (preds > bt).astype(int)
    rep_opt = classification_report(y_te, pred_opt, target_names=['DOWN', 'UP'], output_dict=True)
    lgb_results.append({'name': name, 'auc': auc, 'recall_up_05': rep['UP']['recall'],
                        'f1_up_05': rep['UP']['f1-score'], 'best_thr': bt,
                        'recall_up_opt': rep_opt['UP']['recall'], 'f1_up_opt': rep_opt['UP']['f1-score']})
    print(f"  {name}: AUC={auc:.4f} | Recall UP@0.5={rep['UP']['recall']:.3f} | "
          f"thr={bt:.3f}→Recall={rep_opt['UP']['recall']:.3f}")
    sys.stdout.flush()

print("\n=== 5. Saving results ===")

save_data = {
    'experiment': 'focal_loss_class_imbalance',
    'dl_results': results,
    'lgb_results': lgb_results,
}
with open(DATA / 'features/focal_loss_results.json', 'w') as f:
    json.dump(save_data, f, indent=2, default=str)

print("\n" + "=" * 90)
print(f"{'Model':<25} {'AUC':>6} {'RecUP@0.5':>10} {'F1UP@0.5':>9} {'BestThr':>8} {'RecUP@opt':>10} {'F1UP@opt':>9}")
print("-" * 90)
for r in results:
    print(f"{r['name']:<25} {r['test_auc']:>6.4f} {r['recall_up_05']:>10.3f} {r['f1_up_05']:>9.3f} "
          f"{r['best_threshold']:>8.3f} {r['recall_up_opt']:>10.3f} {r['f1_up_opt']:>9.3f}")
print("-" * 90)
for r in lgb_results:
    print(f"{r['name']:<25} {r['auc']:>6.4f} {r['recall_up_05']:>10.3f} {r['f1_up_05']:>9.3f} "
          f"{r['best_thr']:>8.3f} {r['recall_up_opt']:>10.3f} {r['f1_up_opt']:>9.3f}")
print("=" * 90)
print("\nDone! Results saved to data/features/focal_loss_results.json")
