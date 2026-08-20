import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score,
                             precision_score, recall_score, classification_report)
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
N_FEATURES = 4

print("=" * 70)
print("META-LABELING EXPERIMENT (López de Prado AFML Ch3)")
print("=" * 70)
sys.stdout.flush()

print("\n=== 1. Loading data ===")
sys.stdout.flush()
t0 = time.time()

prices = pd.read_parquet(DATA / 'processed/prices.parquet')
print(f"Prices: {prices.shape}, Tokens: {prices['token_id'].nunique()}")

MIN_SEQ_LEN = WINDOW + HORIZON + 10
sequences_X, sequences_y = [], []
n_skipped = 0

grouped = prices.groupby('token_id')['price']
n_tokens = len(grouped)

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

    current_p = p[np.arange(n_seq) + WINDOW - 1]
    future_p = p[np.arange(n_seq) + WINDOW + HORIZON - 1]
    labels = (future_p > current_p).astype(np.float32)

    valid = ~np.isnan(X_windows.reshape(n_seq, -1)).any(axis=1)
    if valid.sum() > 0:
        sequences_X.append(X_windows[valid])
        sequences_y.append(labels[valid])

X_raw = np.concatenate(sequences_X, axis=0)
y = np.concatenate(sequences_y, axis=0)
del sequences_X, sequences_y

elapsed = time.time() - t0
print(f"Dataset: {X_raw.shape[0]:,} samples, {n_tokens - n_skipped} tokens ({elapsed:.1f}s)")
print(f"Class balance: UP={y.mean():.1%}, DOWN={1-y.mean():.1%}")
sys.stdout.flush()

print("\n=== 2. Feature engineering ===")
sys.stdout.flush()

def make_features(X_windows):
    n = X_windows.shape[0]
    feats = []
    names = []
    for f_idx, f_name in enumerate(['price', 'return', 'volatility', 'momentum']):
        col = X_windows[:, :, f_idx]
        feats.append(col[:, -1]);       names.append(f'{f_name}_last')
        feats.append(col.mean(axis=1)); names.append(f'{f_name}_mean')
        feats.append(col.std(axis=1));  names.append(f'{f_name}_std')
        feats.append(col[:, -1] - col[:, 0]); names.append(f'{f_name}_change')
        feats.append(col[:, -1] - col.mean(axis=1)); names.append(f'{f_name}_dev')

    price_last = X_windows[:, -1, 0]
    feats.append(np.abs(price_last - 0.5)); names.append('price_extremity')
    feats.append((price_last > 0.9).astype(float)); names.append('near_one')
    feats.append((price_last < 0.1).astype(float)); names.append('near_zero')

    vol_last = X_windows[:, -1, 2]
    vol_mean = X_windows[:, :, 2].mean(axis=1)
    feats.append(np.where(vol_mean > 0, vol_last / (vol_mean + 1e-8), 0)); names.append('vol_ratio')
    return np.column_stack(feats), names

X_feat, feat_names = make_features(X_raw)
print(f"Features: {len(feat_names)} ({feat_names[:5]}...)")

n = len(X_feat)
train_end = int(n * 0.60)
val_end = int(n * 0.70)
meta_train_end = int(n * 0.85)

print(f"Split: Primary train={train_end:,} | Meta train={val_end-train_end:,} | "
      f"Meta val={meta_train_end-val_end:,} | Test={n-meta_train_end:,}")
sys.stdout.flush()

print("\n=== 3. Primary model (LGB) ===")
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

X_primary_train = X_feat[:train_end]
y_primary_train = y[:train_end]
X_primary_val = X_feat[train_end:val_end]
y_primary_val = y[train_end:val_end]

dtrain = lgb.Dataset(X_primary_train, y_primary_train, feature_name=feat_names)
dval = lgb.Dataset(X_primary_val, y_primary_val, feature_name=feat_names)

primary_model = lgb.train(
    LGB_PARAMS, dtrain,
    num_boost_round=500,
    valid_sets=[dval],
    callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]
)

primary_pred_all = primary_model.predict(X_feat)
primary_direction = (primary_pred_all > 0.5).astype(int)
primary_correct = (primary_direction == y).astype(float)

test_mask = np.arange(n) >= meta_train_end
auc_primary = roc_auc_score(y[test_mask], primary_pred_all[test_mask])
acc_primary = accuracy_score(y[test_mask], primary_direction[test_mask])
wr_primary = primary_correct[test_mask].mean()

print(f"Primary AUC (test): {auc_primary:.4f}")
print(f"Primary Acc (test): {acc_primary:.4f}")
print(f"Primary WR (test):  {wr_primary:.1%}")
print(f"Primary correct rate by split:")
print(f"  Train:      {primary_correct[:train_end].mean():.1%}")
print(f"  Meta-train: {primary_correct[train_end:val_end].mean():.1%}")
print(f"  Meta-val:   {primary_correct[val_end:meta_train_end].mean():.1%}")
print(f"  Test:       {primary_correct[meta_train_end:].mean():.1%}")
sys.stdout.flush()

print("\n=== 4. Meta-model features ===")
sys.stdout.flush()

def make_meta_features(X_feat, primary_pred, feat_names):
    meta_feats = []
    meta_names = []

    meta_feats.append(primary_pred)
    meta_names.append('primary_prob')

    confidence = np.abs(primary_pred - 0.5) * 2
    meta_feats.append(confidence)
    meta_names.append('primary_confidence')

    meta_feats.append((primary_pred > 0.5).astype(float))
    meta_names.append('primary_direction')

    for i, name in enumerate(feat_names):
        meta_feats.append(X_feat[:, i])
        meta_names.append(f'ctx_{name}')

    price_ext_idx = feat_names.index('price_extremity')
    meta_feats.append(confidence * X_feat[:, price_ext_idx])
    meta_names.append('conf_x_extremity')

    vol_idx = feat_names.index('volatility_last')
    meta_feats.append(confidence * X_feat[:, vol_idx])
    meta_names.append('conf_x_volatility')

    return np.column_stack(meta_feats), meta_names

X_meta, meta_names = make_meta_features(X_feat, primary_pred_all, feat_names)
print(f"Meta features: {len(meta_names)}")
sys.stdout.flush()

print("\n" + "=" * 70)
print("EXPERIMENT 1: Meta-Labeling (predict if primary is correct)")
print("=" * 70)
sys.stdout.flush()

y_meta = primary_correct

X_meta_train = X_meta[train_end:val_end]
y_meta_train = y_meta[train_end:val_end]
X_meta_val = X_meta[val_end:meta_train_end]
y_meta_val = y_meta[val_end:meta_train_end]
X_meta_test = X_meta[meta_train_end:]
y_meta_test = y_meta[meta_train_end:]

print(f"Meta train: {len(X_meta_train):,} (correct rate: {y_meta_train.mean():.1%})")
print(f"Meta val:   {len(X_meta_val):,} (correct rate: {y_meta_val.mean():.1%})")
print(f"Meta test:  {len(X_meta_test):,} (correct rate: {y_meta_test.mean():.1%})")
sys.stdout.flush()

print("\n--- 1a: Meta-model (minimal: primary_prob + confidence + direction) ---")
minimal_cols = [meta_names.index(n) for n in ['primary_prob', 'primary_confidence', 'primary_direction']]
X_mm_tr = X_meta_train[:, minimal_cols]
X_mm_va = X_meta_val[:, minimal_cols]
X_mm_te = X_meta_test[:, minimal_cols]
mm_names = ['primary_prob', 'primary_confidence', 'primary_direction']

dtrain = lgb.Dataset(X_mm_tr, y_meta_train, feature_name=mm_names)
dval = lgb.Dataset(X_mm_va, y_meta_val, feature_name=mm_names)

meta_minimal = lgb.train(
    {**LGB_PARAMS, 'is_unbalance': False},
    dtrain, num_boost_round=500,
    valid_sets=[dval],
    callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]
)

pred_mm = meta_minimal.predict(X_mm_te)
auc_mm = roc_auc_score(y_meta_test, pred_mm)
print(f"  Meta AUC (can it predict correctness?): {auc_mm:.4f}")
sys.stdout.flush()

print("\n--- 1b: Meta-model (full: all context features) ---")
dtrain = lgb.Dataset(X_meta_train, y_meta_train, feature_name=meta_names)
dval = lgb.Dataset(X_meta_val, y_meta_val, feature_name=meta_names)

meta_full = lgb.train(
    {**LGB_PARAMS, 'is_unbalance': False},
    dtrain, num_boost_round=500,
    valid_sets=[dval],
    callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]
)

pred_mf = meta_full.predict(X_meta_test)
auc_mf = roc_auc_score(y_meta_test, pred_mf)
print(f"  Meta AUC (full): {auc_mf:.4f}")

imp = pd.DataFrame({
    'feature': meta_names,
    'importance': meta_full.feature_importance(importance_type='gain')
}).sort_values('importance', ascending=False)
print(f"\n  Top 10 meta-features:")
for _, row in imp.head(10).iterrows():
    print(f"    {row['feature']:30s} {row['importance']:10.1f}")
sys.stdout.flush()

print("\n" + "=" * 70)
print("EXPERIMENT 2: Meta-Filter (skip when P(correct) < threshold)")
print("=" * 70)
sys.stdout.flush()

y_test = y[meta_train_end:]
pred_primary_test = primary_pred_all[meta_train_end:]
direction_test = primary_direction[meta_train_end:]

print(f"\n{'Filter':>20} | {'Kept':>7} | {'%Kept':>6} | {'WR':>6} | {'AUC':>6} | {'PnL/trade':>9}")
print("-" * 75)

wr_nofilter = (direction_test == y_test).mean()

pnl_nofilter = (2 * (direction_test == y_test).astype(float) - 1).mean()
auc_nofilter = roc_auc_score(y_test, pred_primary_test)
print(f"{'No filter':<20} | {len(y_test):>7,} | {1.0:>5.1%} | {wr_nofilter:.4f} | {auc_nofilter:.4f} | {pnl_nofilter:>+8.4f}")

results = {'no_filter': {'n': len(y_test), 'wr': float(wr_nofilter), 'auc': float(auc_nofilter), 'pnl_per_trade': float(pnl_nofilter)}}

for conf_thr in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
    conf = np.abs(pred_primary_test - 0.5)
    mask = conf >= conf_thr
    if mask.sum() < 100:
        continue
    wr = (direction_test[mask] == y_test[mask]).mean()
    pnl = (2 * (direction_test[mask] == y_test[mask]).astype(float) - 1).mean()
    auc = roc_auc_score(y_test[mask], pred_primary_test[mask]) if len(np.unique(y_test[mask])) > 1 else 0
    pct = mask.sum() / len(y_test)
    print(f"{'Conf≥'+str(conf_thr):<20} | {mask.sum():>7,} | {pct:>5.1%} | {wr:.4f} | {auc:.4f} | {pnl:>+8.4f}")
    results[f'conf_{conf_thr}'] = {'n': int(mask.sum()), 'wr': float(wr), 'auc': float(auc), 'pnl_per_trade': float(pnl)}

print()
for meta_thr in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
    mask = pred_mf >= meta_thr
    if mask.sum() < 100:
        continue
    wr = (direction_test[mask] == y_test[mask]).mean()
    pnl = (2 * (direction_test[mask] == y_test[mask]).astype(float) - 1).mean()
    auc = roc_auc_score(y_test[mask], pred_primary_test[mask]) if len(np.unique(y_test[mask])) > 1 else 0
    pct = mask.sum() / len(y_test)
    print(f"{'Meta≥'+str(meta_thr):<20} | {mask.sum():>7,} | {pct:>5.1%} | {wr:.4f} | {auc:.4f} | {pnl:>+8.4f}")
    results[f'meta_{meta_thr}'] = {'n': int(mask.sum()), 'wr': float(wr), 'auc': float(auc), 'pnl_per_trade': float(pnl)}

sys.stdout.flush()

print("\n" + "=" * 70)
print("EXPERIMENT 3: Bet Sizing (P(correct) as weight)")
print("=" * 70)
sys.stdout.flush()

flat_pnl_per_trade = 2 * (direction_test == y_test).astype(float) - 1
flat_total = flat_pnl_per_trade.sum()
flat_sr = flat_pnl_per_trade.mean() / (flat_pnl_per_trade.std() + 1e-8)

conf_weight = np.abs(pred_primary_test - 0.5) * 2
conf_pnl = flat_pnl_per_trade * conf_weight
conf_total = conf_pnl.sum()
conf_sr = conf_pnl.mean() / (conf_pnl.std() + 1e-8)

meta_weight = np.clip(pred_mf, 0, 1)
meta_pnl = flat_pnl_per_trade * meta_weight
meta_total = meta_pnl.sum()
meta_sr = meta_pnl.mean() / (meta_pnl.std() + 1e-8)

meta_mask = pred_mf >= 0.5
if meta_mask.sum() > 0:
    mfw_pnl = flat_pnl_per_trade[meta_mask] * meta_weight[meta_mask]
    mfw_total = mfw_pnl.sum()
    mfw_sr = mfw_pnl.mean() / (mfw_pnl.std() + 1e-8)
    mfw_n = meta_mask.sum()
else:
    mfw_total = mfw_sr = 0
    mfw_n = 0

print(f"\n{'Method':<30} | {'Trades':>7} | {'Total PnL':>10} | {'PnL/trade':>9} | {'Sharpe':>7}")
print("-" * 75)
print(f"{'Flat bet ($1)':<30} | {len(flat_pnl_per_trade):>7,} | {flat_total:>+10.1f} | {flat_pnl_per_trade.mean():>+8.4f} | {flat_sr:>7.4f}")
print(f"{'Confidence-weighted':<30} | {len(conf_pnl):>7,} | {conf_total:>+10.1f} | {conf_pnl.mean():>+8.4f} | {conf_sr:>7.4f}")
print(f"{'Meta-weighted':<30} | {len(meta_pnl):>7,} | {meta_total:>+10.1f} | {meta_pnl.mean():>+8.4f} | {meta_sr:>7.4f}")
print(f"{'Meta-filtered + weighted':<30} | {mfw_n:>7,} | {mfw_total:>+10.1f} | {mfw_pnl.mean() if mfw_n > 0 else 0:>+8.4f} | {mfw_sr:>7.4f}")
sys.stdout.flush()

print("\n" + "=" * 70)
print("EXPERIMENT 4: Analysis of meta-model decisions")
print("=" * 70)
sys.stdout.flush()

high_meta = pred_mf >= 0.6
low_meta = pred_mf < 0.5

price_test = X_feat[meta_train_end:, feat_names.index('price_last')]
vol_test = X_feat[meta_train_end:, feat_names.index('volatility_last')]

print(f"\nMeta says 'trade' (P≥0.6): {high_meta.sum():,} samples")
if high_meta.sum() > 0:
    print(f"  WR: {(direction_test[high_meta] == y_test[high_meta]).mean():.1%}")
    print(f"  Avg price: {price_test[high_meta].mean():.3f}")
    print(f"  Avg volatility: {vol_test[high_meta].mean():.4f}")

print(f"\nMeta says 'skip' (P<0.5): {low_meta.sum():,} samples")
if low_meta.sum() > 0:
    print(f"  WR: {(direction_test[low_meta] == y_test[low_meta]).mean():.1%}")
    print(f"  Avg price: {price_test[low_meta].mean():.3f}")
    print(f"  Avg volatility: {vol_test[low_meta].mean():.4f}")

print(f"\nWR by price level (with/without meta-filter P≥0.6):")
bins = [(0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]
print(f"{'Price bin':>12} | {'All WR':>6} | {'N_all':>7} | {'Meta WR':>7} | {'N_meta':>7} | {'Delta':>7}")
print("-" * 60)
for lo, hi in bins:
    all_mask = (price_test >= lo) & (price_test < hi)
    meta_mask_bin = all_mask & high_meta
    if all_mask.sum() > 0:
        wr_all = (direction_test[all_mask] == y_test[all_mask]).mean()
    else:
        wr_all = 0
    if meta_mask_bin.sum() > 0:
        wr_meta = (direction_test[meta_mask_bin] == y_test[meta_mask_bin]).mean()
    else:
        wr_meta = 0
    delta = wr_meta - wr_all if meta_mask_bin.sum() > 0 and all_mask.sum() > 0 else 0
    print(f"  [{lo:.1f}, {hi:.1f}) | {wr_all:>5.1%} | {all_mask.sum():>7,} | {wr_meta:>6.1%} | {meta_mask_bin.sum():>7,} | {delta:>+6.1%}")

sys.stdout.flush()

print("\n" + "=" * 70)
print("GRAND SUMMARY")
print("=" * 70)

print(f"""
Primary model: AUC={auc_primary:.4f}, WR={wr_primary:.1%}
Meta-model AUC (can predict correctness): minimal={auc_mm:.4f}, full={auc_mf:.4f}

Key findings:""")

print(f"  Meta-model {'CAN' if auc_mf > 0.52 else 'CANNOT'} predict primary correctness (AUC={auc_mf:.4f})")

best_filter = None
best_improvement = 0
for key, val in results.items():
    if key == 'no_filter':
        continue
    imp = val['wr'] - results['no_filter']['wr']
    if imp > best_improvement and val['n'] > 1000:
        best_improvement = imp
        best_filter = key

if best_filter:
    bf = results[best_filter]
    print(f"  Best filter: {best_filter} → WR={bf['wr']:.1%} (vs {results['no_filter']['wr']:.1%} baseline)")
    print(f"    Trades kept: {bf['n']:,} ({bf['n']/results['no_filter']['n']:.0%})")
    print(f"    WR improvement: {best_improvement:+.1%}")
else:
    print(f"  No filter improves WR significantly")

print(f"\n  Bet sizing Sharpe: flat={flat_sr:.4f}, conf-weighted={conf_sr:.4f}, meta-weighted={meta_sr:.4f}")
best_sizing = max([('flat', flat_sr), ('confidence', conf_sr), ('meta', meta_sr)], key=lambda x: x[1])
print(f"  Best sizing: {best_sizing[0]} (SR={best_sizing[1]:.4f})")

sys.stdout.flush()

output = {
    'primary': {'auc': float(auc_primary), 'acc': float(acc_primary), 'wr': float(wr_primary)},
    'meta_auc': {'minimal': float(auc_mm), 'full': float(auc_mf)},
    'filters': results,
    'bet_sizing': {
        'flat': {'total_pnl': float(flat_total), 'sharpe': float(flat_sr)},
        'confidence': {'total_pnl': float(conf_total), 'sharpe': float(conf_sr)},
        'meta': {'total_pnl': float(meta_total), 'sharpe': float(meta_sr)},
        'meta_filtered': {'total_pnl': float(mfw_total), 'sharpe': float(mfw_sr), 'n_trades': int(mfw_n)},
    },
    'config': {'window': WINDOW, 'horizon': HORIZON, 'seed': SEED, 'n_samples': int(n)}
}

output_path = DATA / 'features/meta_labeling_results.json'
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\nResults saved to {output_path}")

total_time = time.time() - t0
print(f"Total time: {total_time/60:.1f} minutes")
sys.stdout.flush()
