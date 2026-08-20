import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score,
                             precision_score, recall_score, classification_report)
from sklearn.linear_model import LinearRegression
from pathlib import Path
import json
import time
import sys

SEED = 42
np.random.seed(SEED)

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / 'data'

WINDOW = 48
HORIZON = 12
HORIZONS_SCAN = [3, 6, 12, 18, 24, 36, 48]
BACK_HORIZONS = [6, 12, 24, 48]

print("=" * 70)
print("TREND-SCANNING EXPERIMENT (López de Prado Ch5)")
print("=" * 70)
sys.stdout.flush()

def vectorized_tval(prices_2d, L):
    n = prices_2d.shape[0]
    x = np.arange(L, dtype=np.float64)
    x_mean = x.mean()
    SXX = ((x - x_mean) ** 2).sum()

    y_mean = prices_2d.mean(axis=1, keepdims=True)
    x_centered = (x - x_mean)[np.newaxis, :]

    SXY = (x_centered * (prices_2d - y_mean)).sum(axis=1)
    beta1 = SXY / SXX

    y_hat = y_mean + beta1[:, np.newaxis] * x_centered
    SSR = ((prices_2d - y_hat) ** 2).sum(axis=1)

    if L <= 2:
        return np.zeros(n)

    sigma2 = SSR / (L - 2)
    SE_beta1 = np.sqrt(np.maximum(sigma2 / SXX, 1e-30))

    t_val = beta1 / SE_beta1
    return t_val

def trend_scan_forward(prices, horizons):
    n = len(prices)
    max_L = max(horizons)

    best_tval = np.zeros(n)
    best_L = np.zeros(n, dtype=int)
    best_dir = np.zeros(n)
    valid_mask = np.zeros(n, dtype=bool)

    p = prices.astype(np.float64)

    for L in horizons:
        if n < L + 1:
            continue
        n_windows = n - L + 1

        windows = np.lib.stride_tricks.as_strided(
            p, shape=(n_windows, L),
            strides=(p.strides[0], p.strides[0])
        ).copy()

        tvals = vectorized_tval(windows, L)

        for i in range(n_windows):
            if abs(tvals[i]) > abs(best_tval[i]):
                best_tval[i] = tvals[i]
                best_L[i] = L
                best_dir[i] = np.sign(tvals[i])
                valid_mask[i] = True

    return best_tval, best_L, best_dir, valid_mask

def trend_features_backward(prices, horizons):
    n = len(prices)
    p = prices.astype(np.float64)

    features = {}

    for L in horizons:
        if n < L + 1:
            tval = np.full(n, np.nan)
            r2 = np.full(n, np.nan)
        else:

            n_windows = n - L + 1
            windows = np.lib.stride_tricks.as_strided(
                p, shape=(n_windows, L),
                strides=(p.strides[0], p.strides[0])
            ).copy()

            tvals = vectorized_tval(windows, L)

            x = np.arange(L, dtype=np.float64)
            x_mean = x.mean()
            SXX = ((x - x_mean) ** 2).sum()
            y_mean = windows.mean(axis=1, keepdims=True)
            SXY = ((x - x_mean)[np.newaxis, :] * (windows - y_mean)).sum(axis=1)
            beta1 = SXY / SXX
            y_hat = y_mean + beta1[:, np.newaxis] * (x - x_mean)[np.newaxis, :]
            SS_res = ((windows - y_hat) ** 2).sum(axis=1)
            SS_tot = ((windows - y_mean) ** 2).sum(axis=1)
            r2_vals = np.where(SS_tot > 1e-12, 1 - SS_res / SS_tot, 0.0)

            tval = np.full(n, np.nan)
            tval[L-1:] = tvals
            r2 = np.full(n, np.nan)
            r2[L-1:] = r2_vals

        features[f'trend_tval_{L}h'] = tval
        features[f'trend_r2_{L}h'] = r2

    tval_matrix = np.column_stack([features[f'trend_tval_{L}h'] for L in horizons])
    abs_tvals = np.abs(tval_matrix)
    abs_tvals = np.where(np.isnan(abs_tvals), -1, abs_tvals)
    best_idx = abs_tvals.argmax(axis=1)

    best_tval = np.array([tval_matrix[i, best_idx[i]] if abs_tvals[i, best_idx[i]] >= 0 else np.nan
                          for i in range(n)])
    best_L = np.array([horizons[best_idx[i]] if abs_tvals[i, best_idx[i]] >= 0 else 0
                       for i in range(n)])

    features['trend_best_tval'] = best_tval
    features['trend_best_L'] = best_L.astype(float)
    features['trend_strength'] = np.abs(best_tval)

    return features

print("\n=== 1. Loading data ===")
sys.stdout.flush()

prices = pd.read_parquet(DATA / 'processed/prices.parquet')
print(f"Prices: {prices.shape}, Tokens: {prices['token_id'].nunique()}")
sys.stdout.flush()

MIN_SEQ_LEN = WINDOW + max(HORIZONS_SCAN) + 10
N_BASE_FEATURES = 4

t0 = time.time()

all_X_base = []
all_X_trend = []
all_y_fixed = []
all_y_adaptive = []
all_tval_fwd = []
all_best_L_fwd = []
n_skipped = 0

grouped = prices.groupby('token_id')['price']
n_tokens = len(grouped)
print(f"Processing {n_tokens} tokens (trend-scanning + features)...")
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
    base_features = np.column_stack([p, ret, vol, momentum])

    trend_feats = trend_features_backward(p, BACK_HORIZONS)
    trend_arr = np.column_stack([trend_feats[k] for k in sorted(trend_feats.keys())])

    fwd_tval, fwd_L, fwd_dir, fwd_valid = trend_scan_forward(p, HORIZONS_SCAN)

    T = len(p)
    n_seq = T - WINDOW - max(HORIZONS_SCAN)
    if n_seq <= 0:
        n_skipped += 1
        continue

    strides = base_features.strides
    X_base_windows = np.lib.stride_tricks.as_strided(
        base_features,
        shape=(n_seq, WINDOW, N_BASE_FEATURES),
        strides=(strides[0], strides[0], strides[1])
    ).copy()

    current_positions = np.arange(n_seq) + WINDOW - 1
    X_trend_current = trend_arr[current_positions]

    current_p = p[current_positions]
    future_p_12h = p[current_positions + HORIZON]
    y_fixed = (future_p_12h > current_p).astype(np.float32)

    y_adaptive = np.where(fwd_dir[current_positions] > 0, 1.0, 0.0).astype(np.float32)
    tval_fwd = fwd_tval[current_positions]
    best_L_fwd_arr = fwd_L[current_positions]

    valid_base = ~np.isnan(X_base_windows.reshape(n_seq, -1)).any(axis=1)
    valid_trend = ~np.isnan(X_trend_current).any(axis=1)
    valid = valid_base & valid_trend & fwd_valid[current_positions]

    if valid.sum() > 0:
        all_X_base.append(X_base_windows[valid])
        all_X_trend.append(X_trend_current[valid])
        all_y_fixed.append(y_fixed[valid])
        all_y_adaptive.append(y_adaptive[valid])
        all_tval_fwd.append(tval_fwd[valid])
        all_best_L_fwd.append(best_L_fwd_arr[valid])

    if (idx + 1) % 100 == 0:
        total = sum(len(s) for s in all_y_fixed)
        elapsed = time.time() - t0
        print(f"  {idx+1}/{n_tokens} tokens | {total:,} sequences | {elapsed:.0f}s")
        sys.stdout.flush()

print("Concatenating...")
sys.stdout.flush()

X_base = np.concatenate(all_X_base, axis=0)
X_trend = np.concatenate(all_X_trend, axis=0)
y_fixed = np.concatenate(all_y_fixed, axis=0)
y_adaptive = np.concatenate(all_y_adaptive, axis=0)
tval_fwd = np.concatenate(all_tval_fwd, axis=0)
best_L_fwd = np.concatenate(all_best_L_fwd, axis=0)

del all_X_base, all_X_trend, all_y_fixed, all_y_adaptive, all_tval_fwd, all_best_L_fwd

elapsed = time.time() - t0
print(f"\nDataset: X_base={X_base.shape}, X_trend={X_trend.shape}")
print(f"Labels: y_fixed={y_fixed.shape}, y_adaptive={y_adaptive.shape}")
print(f"Tokens: {n_tokens - n_skipped}/{n_tokens} used ({elapsed:.1f}s)")
print(f"Fixed labels:    UP={y_fixed.mean():.1%}, DOWN={1-y_fixed.mean():.1%}")
print(f"Adaptive labels: UP={y_adaptive.mean():.1%}, DOWN={1-y_adaptive.mean():.1%}")
print(f"Agreement fixed vs adaptive: {(y_fixed == y_adaptive).mean():.1%}")
sys.stdout.flush()

print(f"\nForward scan stats:")
print(f"  |t_val| mean={np.abs(tval_fwd).mean():.2f}, median={np.median(np.abs(tval_fwd)):.2f}")
print(f"  Best L distribution: {dict(zip(*np.unique(best_L_fwd, return_counts=True)))}")
sys.stdout.flush()

print("\n=== 2. Prepare LGB features ===")
sys.stdout.flush()

def summarize_window(X_windows):
    n = X_windows.shape[0]
    feats = []
    names = []

    for f_idx, f_name in enumerate(['price', 'return', 'volatility', 'momentum']):
        col = X_windows[:, :, f_idx]
        feats.append(col[:, -1])
        names.append(f'{f_name}_last')
        feats.append(col.mean(axis=1))
        names.append(f'{f_name}_mean')
        feats.append(col.std(axis=1))
        names.append(f'{f_name}_std')
        feats.append(col[:, -1] - col[:, 0])
        names.append(f'{f_name}_change')
        feats.append(col[:, -1] - col.mean(axis=1))
        names.append(f'{f_name}_dev')

    return np.column_stack(feats), names

X_summary, summary_names = summarize_window(X_base)
print(f"Summary features: {len(summary_names)} ({summary_names[:5]}...)")

trend_names = sorted(trend_feats.keys())
print(f"Trend features: {len(trend_names)} ({trend_names[:5]}...)")

X_combined = np.hstack([X_summary, X_trend])
all_names = summary_names + trend_names
print(f"Combined: {X_combined.shape[1]} features")
sys.stdout.flush()

n = len(X_combined)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

def split_data(X, y):
    return (X[:train_end], y[:train_end],
            X[train_end:val_end], y[train_end:val_end],
            X[val_end:], y[val_end:])

print(f"\nSplit: Train={train_end:,} | Val={val_end-train_end:,} | Test={n-val_end:,}")
print(f"Train UP: {y_fixed[:train_end].mean():.1%} | Test UP: {y_fixed[val_end:].mean():.1%}")
sys.stdout.flush()

print("\n" + "=" * 70)
print("EXPERIMENT 1: LGB + Trend Features")
print("=" * 70)
sys.stdout.flush()

LGB_PARAMS = {
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': 6,
    'min_child_samples': 50,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'is_unbalance': True,
    'random_state': SEED,
    'verbose': -1,
    'n_jobs': -1,
}

results = {}

print("\n--- 1a: LGB Baseline (20 summary features, fixed 12h labels) ---")
sys.stdout.flush()
X_tr, y_tr, X_va, y_va, X_te, y_te = split_data(X_summary, y_fixed)

dtrain = lgb.Dataset(X_tr, y_tr, feature_name=summary_names)
dval = lgb.Dataset(X_va, y_va, feature_name=summary_names)

model_base = lgb.train(
    LGB_PARAMS, dtrain,
    num_boost_round=500,
    valid_sets=[dval],
    callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]
)

pred_base = model_base.predict(X_te)
auc_base = roc_auc_score(y_te, pred_base)
pred_binary = (pred_base > 0.5).astype(int)
acc_base = accuracy_score(y_te, pred_binary)
rec_base = recall_score(y_te, pred_binary)
f1_base = f1_score(y_te, pred_binary)

print(f"  AUC={auc_base:.4f}, Acc={acc_base:.4f}, RecUP={rec_base:.4f}, F1UP={f1_base:.4f}")
print(f"  Best iter: {model_base.best_iteration}")
results['lgb_baseline'] = {'auc': auc_base, 'acc': acc_base, 'rec_up': rec_base, 'f1_up': f1_base}
sys.stdout.flush()

print("\n--- 1b: LGB + Trend Features (31 features, fixed 12h labels) ---")
sys.stdout.flush()
X_tr, y_tr, X_va, y_va, X_te, y_te = split_data(X_combined, y_fixed)

dtrain = lgb.Dataset(X_tr, y_tr, feature_name=all_names)
dval = lgb.Dataset(X_va, y_va, feature_name=all_names)

model_trend = lgb.train(
    LGB_PARAMS, dtrain,
    num_boost_round=500,
    valid_sets=[dval],
    callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]
)

pred_trend = model_trend.predict(X_te)
auc_trend = roc_auc_score(y_te, pred_trend)
pred_binary = (pred_trend > 0.5).astype(int)
acc_trend = accuracy_score(y_te, pred_binary)
rec_trend = recall_score(y_te, pred_binary)
f1_trend = f1_score(y_te, pred_binary)

print(f"  AUC={auc_trend:.4f}, Acc={acc_trend:.4f}, RecUP={rec_trend:.4f}, F1UP={f1_trend:.4f}")
print(f"  Best iter: {model_trend.best_iteration}")
print(f"  AUC improvement: {auc_trend - auc_base:+.4f}")
results['lgb_trend_features'] = {'auc': auc_trend, 'acc': acc_trend, 'rec_up': rec_trend, 'f1_up': f1_trend}
sys.stdout.flush()

imp = pd.DataFrame({
    'feature': all_names,
    'importance': model_trend.feature_importance(importance_type='gain')
}).sort_values('importance', ascending=False)
print(f"\n  Top 10 features (gain):")
for _, row in imp.head(10).iterrows():
    marker = " ★" if 'trend' in row['feature'] else ""
    print(f"    {row['feature']:25s} {row['importance']:10.1f}{marker}")
sys.stdout.flush()

print("\n" + "=" * 70)
print("EXPERIMENT 2: Adaptive Labels (trend-scanning) vs Fixed 12h")
print("=" * 70)
sys.stdout.flush()

print("\n--- 2a: LGB Baseline features + ADAPTIVE labels ---")
sys.stdout.flush()
X_tr, y_tr, X_va, y_va, X_te, y_te_adaptive = split_data(X_summary, y_adaptive)

dtrain = lgb.Dataset(X_tr, y_tr, feature_name=summary_names)
dval = lgb.Dataset(X_va, y_va, feature_name=summary_names)

model_adaptive = lgb.train(
    LGB_PARAMS, dtrain,
    num_boost_round=500,
    valid_sets=[dval],
    callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]
)

pred_adaptive = model_adaptive.predict(X_te)

auc_adapt_self = roc_auc_score(y_te_adaptive, pred_adaptive)

y_te_fixed = y_fixed[val_end:]
auc_adapt_vs_fixed = roc_auc_score(y_te_fixed, pred_adaptive)

print(f"  AUC vs adaptive labels: {auc_adapt_self:.4f}")
print(f"  AUC vs fixed 12h labels: {auc_adapt_vs_fixed:.4f}")
print(f"  Vs baseline (fixed→fixed): {auc_adapt_vs_fixed - auc_base:+.4f}")
results['lgb_adaptive_labels'] = {
    'auc_self': auc_adapt_self,
    'auc_vs_fixed': auc_adapt_vs_fixed,
}
sys.stdout.flush()

print("\n--- 2b: LGB + Trend features + ADAPTIVE labels ---")
sys.stdout.flush()
X_tr, y_tr, X_va, y_va, X_te, y_te_adaptive = split_data(X_combined, y_adaptive)

dtrain = lgb.Dataset(X_tr, y_tr, feature_name=all_names)
dval = lgb.Dataset(X_va, y_va, feature_name=all_names)

model_adapt_trend = lgb.train(
    LGB_PARAMS, dtrain,
    num_boost_round=500,
    valid_sets=[dval],
    callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]
)

pred_at = model_adapt_trend.predict(X_te)
auc_at_self = roc_auc_score(y_te_adaptive, pred_at)
auc_at_vs_fixed = roc_auc_score(y_te_fixed, pred_at)

print(f"  AUC vs adaptive labels: {auc_at_self:.4f}")
print(f"  AUC vs fixed 12h labels: {auc_at_vs_fixed:.4f}")
print(f"  Vs baseline (fixed→fixed): {auc_at_vs_fixed - auc_base:+.4f}")
results['lgb_adaptive_trend'] = {
    'auc_self': auc_at_self,
    'auc_vs_fixed': auc_at_vs_fixed,
}
sys.stdout.flush()

print("\n" + "=" * 70)
print("EXPERIMENT 3: Trend Filter (skip low |t_val|)")
print("=" * 70)
sys.stdout.flush()

tval_test = tval_fwd[val_end:]
best_pred = pred_trend
y_te_fixed = y_fixed[val_end:]

thresholds = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
print(f"\n{'Threshold':>10} | {'Samples':>8} | {'%Kept':>6} | {'AUC':>6} | {'Acc':>6} | {'RecUP':>6} | {'F1UP':>6}")
print("-" * 72)

filter_results = []
for thr in thresholds:
    mask = np.abs(tval_test) >= thr
    n_kept = mask.sum()
    if n_kept < 100:
        print(f"  |t_val|≥{thr}: only {n_kept} samples, skipping")
        continue

    pct_kept = n_kept / len(tval_test)
    y_filt = y_te_fixed[mask]
    pred_filt = best_pred[mask]

    auc_f = roc_auc_score(y_filt, pred_filt)
    pred_bin = (pred_filt > 0.5).astype(int)
    acc_f = accuracy_score(y_filt, pred_bin)
    rec_f = recall_score(y_filt, pred_bin, zero_division=0)
    f1_f = f1_score(y_filt, pred_bin, zero_division=0)

    filter_results.append({
        'threshold': thr, 'n_samples': int(n_kept), 'pct_kept': pct_kept,
        'auc': auc_f, 'acc': acc_f, 'rec_up': rec_f, 'f1_up': f1_f
    })

    print(f"  |t|≥{thr:<4.1f} | {n_kept:>8,} | {pct_kept:>5.1%} | {auc_f:.4f} | {acc_f:.4f} | {rec_f:.4f} | {f1_f:.4f}")
    sys.stdout.flush()

results['filter_by_tval'] = filter_results

print("\n" + "=" * 70)
print("EXPERIMENT 4: Which horizon works best?")
print("=" * 70)
sys.stdout.flush()

print(f"\n{'Horizon':>8} | {'Train UP%':>9} | {'Test UP%':>8} | {'AUC':>6} | {'Acc':>6}")
print("-" * 50)

horizon_results = []
for H in HORIZONS_SCAN:

    current_positions_test = np.arange(n - val_end) + WINDOW - 1

    n_full = len(X_base)
    cp = X_base[:, -1, 0]

    pass

print(f"\nBest forward horizon distribution (test set):")
L_test = best_L_fwd[val_end:]
for L in sorted(HORIZONS_SCAN):
    count = (L_test == L).sum()
    pct = count / len(L_test)

    mask = L_test == L
    mean_tval = np.abs(tval_test[mask]).mean() if mask.sum() > 0 else 0
    print(f"  L={L:2d}h: {count:>7,} ({pct:>5.1%}) | mean |t_val|={mean_tval:.2f}")
sys.stdout.flush()

print(f"\nLabel agreement (adaptive vs fixed 12h) by best_L:")
for L in sorted(HORIZONS_SCAN):
    mask = L_test == L
    if mask.sum() == 0:
        continue
    agree = (y_fixed[val_end:][mask] == y_adaptive[val_end:][mask]).mean()
    print(f"  L={L:2d}h: {agree:.1%} agreement with fixed 12h")
sys.stdout.flush()

print("\n" + "=" * 70)
print("GRAND SUMMARY")
print("=" * 70)

print(f"\n{'Model':<35} | {'AUC':>6} | {'Acc':>6} | {'RecUP':>6}")
print("-" * 62)
print(f"{'LGB baseline (20 feats, fixed)':<35} | {results['lgb_baseline']['auc']:.4f} | {results['lgb_baseline']['acc']:.4f} | {results['lgb_baseline']['rec_up']:.4f}")
print(f"{'LGB + trend (31 feats, fixed)':<35} | {results['lgb_trend_features']['auc']:.4f} | {results['lgb_trend_features']['acc']:.4f} | {results['lgb_trend_features']['rec_up']:.4f}")
print(f"{'LGB baseline + adaptive labels':<35} | {results['lgb_adaptive_labels']['auc_vs_fixed']:.4f} |   —   |   —  ")
print(f"{'LGB + trend + adaptive labels':<35} | {results['lgb_adaptive_trend']['auc_vs_fixed']:.4f} |   —   |   —  ")

if filter_results:
    best_filter = max(filter_results, key=lambda x: x['auc'])
    print(f"\nBest filter: |t_val|≥{best_filter['threshold']:.1f} → AUC={best_filter['auc']:.4f} ({best_filter['pct_kept']:.0%} kept)")

print(f"\nTrend features AUC delta: {results['lgb_trend_features']['auc'] - results['lgb_baseline']['auc']:+.4f}")
print(f"Adaptive labels AUC delta: {results['lgb_adaptive_labels']['auc_vs_fixed'] - results['lgb_baseline']['auc']:+.4f}")
sys.stdout.flush()

output = {
    'experiments': results,
    'filter_by_tval': filter_results,
    'config': {
        'window': WINDOW, 'fixed_horizon': HORIZON,
        'horizons_scan': HORIZONS_SCAN, 'back_horizons': BACK_HORIZONS,
        'seed': SEED, 'n_samples': int(n), 'n_tokens': int(n_tokens - n_skipped),
    }
}

def convert_numpy(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    return obj

output_path = DATA / 'features/trend_scanning_results.json'
with open(output_path, 'w') as f:
    json.dump(convert_numpy(output), f, indent=2)
print(f"\nResults saved to {output_path}")

total_time = time.time() - t0
print(f"Total time: {total_time/60:.1f} minutes")
sys.stdout.flush()
