import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from pathlib import Path
import json
import time
import sys
import warnings
warnings.filterwarnings('ignore')

import torch
torch.set_num_threads(1)

from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS, NBEATSx, DeepAR, PatchTST, BiTCN
from neuralforecast.losses.pytorch import DistributionLoss

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / 'data'

WINDOW = 48
HORIZON = 12

print("=" * 70)
print("NEURALFORECAST EXPERIMENT — Phase 5.5")
print("=" * 70)
sys.stdout.flush()

print("\n=== 1. Loading and preparing data ===")
t0 = time.time()

prices = pd.read_parquet(DATA / 'processed/prices.parquet')
print(f"Raw prices: {prices.shape}, Tokens: {prices['token_id'].nunique()}")

MIN_LEN = WINDOW + HORIZON + 20

dfs = []
token_count = 0
skipped = 0

grouped = prices.groupby('token_id')
for token, group in grouped:
    p = group['price'].values.astype(np.float64)
    if len(p) < MIN_LEN:
        skipped += 1
        continue

    ret = np.zeros_like(p); ret[1:] = np.diff(p)
    vol = pd.Series(ret).rolling(12, min_periods=1).std().values
    ma12 = pd.Series(p).rolling(12, min_periods=1).mean().values
    momentum = p - ma12

    df_token = pd.DataFrame({
        'unique_id': str(token),
        'ds': pd.date_range('2024-01-01', periods=len(p), freq='h'),
        'y': p,
        'return': ret,
        'volatility': vol,
        'momentum': momentum,
    })

    df_token = df_token.dropna()
    if len(df_token) >= MIN_LEN:
        dfs.append(df_token)
        token_count += 1

    if token_count >= 200:
        break

df_all = pd.concat(dfs, ignore_index=True)
n_total = len(df_all)
n_tokens = df_all['unique_id'].nunique()
print(f"Prepared: {n_total:,} rows, {n_tokens} tokens (skipped {skipped})")
print(f"Time: {time.time()-t0:.1f}s")
sys.stdout.flush()

print("\n=== 2. Time-based split ===")

all_dates = sorted(df_all['ds'].unique())
n_dates = len(all_dates)
train_date = all_dates[int(n_dates * 0.70)]
val_date = all_dates[int(n_dates * 0.85)]

df_train = df_all[df_all['ds'] < train_date].copy()
df_val = df_all[(df_all['ds'] >= train_date) & (df_all['ds'] < val_date)].copy()
df_test = df_all[df_all['ds'] >= val_date].copy()

print(f"Train: {len(df_train):,} rows, until {train_date}")
print(f"Val:   {len(df_val):,} rows")
print(f"Test:  {len(df_test):,} rows, from {val_date}")

df_trainval = pd.concat([df_train, df_val], ignore_index=True)
print(f"TrainVal: {len(df_trainval):,} rows")
sys.stdout.flush()

print("\n=== 3. LightGBM baseline ===")
t1 = time.time()

def make_lgb_data(df_all, window=WINDOW, horizon=HORIZON):
    all_X, all_y = [], []

    for token in df_all['unique_id'].unique():
        df_t = df_all[df_all['unique_id'] == token].sort_values('ds')
        p = df_t['y'].values
        ret = df_t['return'].values
        vol = df_t['volatility'].values
        mom = df_t['momentum'].values

        features = np.column_stack([p, ret, vol, mom])
        n_seq = len(p) - window - horizon
        if n_seq <= 0:
            continue

        strides = features.strides
        X_w = np.lib.stride_tricks.as_strided(
            features, shape=(n_seq, window, 4),
            strides=(strides[0], strides[0], strides[1])
        ).copy()

        current_p = p[np.arange(n_seq) + window - 1]
        future_p = p[np.arange(n_seq) + window + horizon - 1]
        labels = (future_p > current_p).astype(np.float32)

        feats = []
        for f_idx in range(4):
            col = X_w[:, :, f_idx]
            feats.extend([col[:, -1], col.mean(axis=1), col.std(axis=1),
                         col[:, -1] - col[:, 0], col[:, -1] - col.mean(axis=1)])

        price_last = X_w[:, -1, 0]
        feats.append(np.abs(price_last - 0.5))
        feats.append((price_last > 0.9).astype(float))
        feats.append((price_last < 0.1).astype(float))
        vol_last = X_w[:, -1, 2]
        vol_mean = X_w[:, :, 2].mean(axis=1)
        feats.append(np.where(vol_mean > 0, vol_last / (vol_mean + 1e-8), 0))

        X = np.column_stack(feats)
        valid = ~np.isnan(X).any(axis=1)
        all_X.append(X[valid])
        all_y.append(labels[valid])

    return np.concatenate(all_X), np.concatenate(all_y)

X_lgb, y_lgb = make_lgb_data(df_all)
n_lgb = len(y_lgb)
train_end = int(n_lgb * 0.70)
val_end = int(n_lgb * 0.85)

feat_names_lgb = []
for name in ['price', 'return', 'volatility', 'momentum']:
    for stat in ['last', 'mean', 'std', 'change', 'dev']:
        feat_names_lgb.append(f'{name}_{stat}')
feat_names_lgb.extend(['price_extremity', 'near_one', 'near_zero', 'vol_ratio'])

dt = lgb.Dataset(X_lgb[:train_end], y_lgb[:train_end], feature_name=feat_names_lgb)
dv = lgb.Dataset(X_lgb[train_end:val_end], y_lgb[train_end:val_end], feature_name=feat_names_lgb)

lgb_model = lgb.train(
    {'objective': 'binary', 'metric': 'auc', 'learning_rate': 0.05,
     'num_leaves': 31, 'max_depth': 6, 'min_child_samples': 50,
     'subsample': 0.8, 'colsample_bytree': 0.8, 'is_unbalance': True,
     'random_state': SEED, 'verbose': -1, 'n_jobs': -1},
    dt, num_boost_round=500, valid_sets=[dv],
    callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]
)

lgb_pred = lgb_model.predict(X_lgb[val_end:])
lgb_auc = roc_auc_score(y_lgb[val_end:], lgb_pred)
lgb_wr = ((lgb_pred > 0.5).astype(int) == y_lgb[val_end:]).mean()
print(f"LightGBM: AUC={lgb_auc:.4f}, WR={lgb_wr:.1%} ({time.time()-t1:.1f}s)")
print(f"  Test samples: {len(y_lgb[val_end:]):,}, UP={y_lgb[val_end:].mean():.1%}")
sys.stdout.flush()

print("\n=== 4. NeuralForecast models ===")
sys.stdout.flush()

hist_exog = ['return', 'volatility', 'momentum']

models_config = [
    ('NHITS', NHITS(
        h=HORIZON,
        input_size=WINDOW,
        max_steps=300,
        learning_rate=1e-3,
        scaler_type='standard',
        random_seed=SEED,
        accelerator='cpu',
        hist_exog_list=hist_exog,
    )),
    ('PatchTST', PatchTST(
        h=HORIZON,
        input_size=WINDOW,
        max_steps=300,
        learning_rate=1e-3,
        scaler_type='standard',
        random_seed=SEED,
        accelerator='cpu',

    )),
    ('BiTCN', BiTCN(
        h=HORIZON,
        input_size=WINDOW,
        max_steps=300,
        learning_rate=1e-3,
        scaler_type='standard',
        random_seed=SEED,
        accelerator='cpu',
        hist_exog_list=hist_exog,
    )),
]

results = {
    'lgb_baseline': {
        'auc': float(lgb_auc),
        'wr': float(lgb_wr),
        'n_test': int(len(y_lgb[val_end:])),
    }
}

for model_name, model in models_config:
    print(f"\n--- {model_name} ---")
    t_model = time.time()
    sys.stdout.flush()

    try:
        nf = NeuralForecast(models=[model], freq='h')

        nf.fit(df=df_trainval, val_size=int(len(df_val) / n_tokens))

        forecasts = nf.cross_validation(
            df=df_all,
            n_windows=1,
            step_size=HORIZON,
        )

        model_col = [c for c in forecasts.columns if model_name in c and 'lo' not in c and 'hi' not in c]
        if not model_col:
            model_col = [c for c in forecasts.columns if c not in ['unique_id', 'ds', 'cutoff', 'y']]

        if model_col:
            col = model_col[0]

            forecast_vals = forecasts[col].values
            actual_vals = forecasts['y'].values

            correct_predictions = 0
            total_predictions = 0
            all_pred_probs = []
            all_actual_dirs = []

            for uid in forecasts['unique_id'].unique():
                uid_forecasts = forecasts[forecasts['unique_id'] == uid].sort_values('ds')
                uid_data = df_all[df_all['unique_id'] == uid].sort_values('ds')

                for _, row in uid_forecasts.iterrows():
                    forecast_ds = row['ds']
                    actual_future = row['y']
                    forecast_price = row[col]

                    current_ds = forecast_ds - pd.Timedelta(hours=HORIZON)
                    current_row = uid_data[uid_data['ds'] == current_ds]

                    if len(current_row) == 0:
                        continue

                    current_price = current_row['y'].values[0]

                    pred_up = float(forecast_price > current_price)
                    actual_up = float(actual_future > current_price)

                    all_pred_probs.append(forecast_price - current_price)
                    all_actual_dirs.append(actual_up)
                    total_predictions += 1

            if total_predictions > 100:
                pred_arr = np.array(all_pred_probs)
                actual_arr = np.array(all_actual_dirs)

                try:
                    auc = roc_auc_score(actual_arr, pred_arr)
                except:
                    auc = 0.5
                wr = ((pred_arr > 0) == actual_arr).mean()

                elapsed = time.time() - t_model
                print(f"  AUC={auc:.4f}, WR={wr:.1%}, N={total_predictions:,} ({elapsed:.0f}s)")

                results[model_name] = {
                    'auc': float(auc),
                    'wr': float(wr),
                    'n_test': total_predictions,
                    'time': float(elapsed),
                }
            else:
                print(f"  Too few predictions: {total_predictions}")
                results[model_name] = {'auc': 0.5, 'wr': 0.5, 'n_test': total_predictions, 'error': 'too_few'}

        else:
            print(f"  No forecast column found: {forecasts.columns.tolist()}")
            results[model_name] = {'error': 'no_column'}

    except Exception as e:
        elapsed = time.time() - t_model
        print(f"  ERROR: {e} ({elapsed:.0f}s)")
        results[model_name] = {'error': str(e), 'time': float(elapsed)}

    sys.stdout.flush()

print("\n=== 5. Bernoulli loss models (direct classification) ===")
sys.stdout.flush()

dfs_binary = []
for token in df_all['unique_id'].unique():
    df_t = df_all[df_all['unique_id'] == token].sort_values('ds').copy()
    p = df_t['y'].values

    future_p = np.full(len(p), np.nan)
    future_p[:len(p)-HORIZON] = p[HORIZON:]
    direction = (future_p > p).astype(float)
    direction[np.isnan(future_p)] = np.nan

    df_t['y'] = direction
    df_t = df_t.dropna(subset=['y'])

    if len(df_t) >= MIN_LEN:
        dfs_binary.append(df_t)

df_binary = pd.concat(dfs_binary, ignore_index=True)
print(f"Binary dataset: {len(df_binary):,} rows, UP={df_binary['y'].mean():.1%}")

all_dates_b = sorted(df_binary['ds'].unique())
train_date_b = all_dates_b[int(len(all_dates_b) * 0.70)]
val_date_b = all_dates_b[int(len(all_dates_b) * 0.85)]
df_trainval_b = df_binary[df_binary['ds'] < val_date_b].copy()

bernoulli_models = [
    ('NHITS_Bernoulli', NHITS(
        h=HORIZON,
        input_size=WINDOW,
        max_steps=300,
        learning_rate=1e-3,
        loss=DistributionLoss(distribution='Bernoulli'),
        scaler_type='standard',
        random_seed=SEED,
        accelerator='cpu',
        hist_exog_list=hist_exog,
    )),
]

for model_name, model in bernoulli_models:
    print(f"\n--- {model_name} ---")
    t_model = time.time()
    sys.stdout.flush()

    try:
        nf = NeuralForecast(models=[model], freq='h')
        n_tokens_b = df_binary['unique_id'].nunique()
        nf.fit(df=df_trainval_b, val_size=max(1, int(len(df_binary[df_binary['ds'] >= train_date_b]) / n_tokens_b / 2)))

        forecasts = nf.cross_validation(df=df_binary, n_windows=1, step_size=HORIZON)

        model_col = [c for c in forecasts.columns if 'NHITS' in c and 'lo' not in c and 'hi' not in c]
        if not model_col:
            model_col = [c for c in forecasts.columns if c not in ['unique_id', 'ds', 'cutoff', 'y']]

        if model_col:
            col = model_col[0]
            pred = forecasts[col].values
            actual = forecasts['y'].values

            valid = ~(np.isnan(pred) | np.isnan(actual))
            pred = pred[valid]
            actual = actual[valid]

            if len(pred) > 100:
                try:
                    auc = roc_auc_score(actual, pred)
                except:
                    auc = 0.5
                wr = ((pred > 0.5) == actual).mean()

                elapsed = time.time() - t_model
                print(f"  AUC={auc:.4f}, WR={wr:.1%}, N={len(pred):,} ({elapsed:.0f}s)")
                results[model_name] = {'auc': float(auc), 'wr': float(wr), 'n_test': int(len(pred)), 'time': float(elapsed)}
            else:
                print(f"  Too few valid predictions: {len(pred)}")
                results[model_name] = {'auc': 0.5, 'error': 'too_few'}
        else:
            print(f"  No column found: {forecasts.columns.tolist()}")
            results[model_name] = {'error': 'no_column'}

    except Exception as e:
        elapsed = time.time() - t_model
        print(f"  ERROR: {e} ({elapsed:.0f}s)")
        results[model_name] = {'error': str(e), 'time': float(elapsed)}

    sys.stdout.flush()

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"\n{'Model':<25} | {'AUC':>7} | {'WR':>6} | {'N_test':>8} | {'Time':>6}")
print("-" * 62)

for name, res in sorted(results.items(), key=lambda x: x[1].get('auc', 0), reverse=True):
    if 'error' in res and 'auc' not in res:
        print(f"{name:<25} | {'ERROR':>7} | {'—':>6} | {'—':>8} | {'—':>6}")
    else:
        auc = res.get('auc', 0)
        wr = res.get('wr', 0)
        n = res.get('n_test', 0)
        t = res.get('time', 0)
        marker = ' ← BEST' if name == 'lgb_baseline' else ''
        print(f"{name:<25} | {auc:>6.4f} | {wr:>5.1%} | {n:>8,} | {t:>5.0f}s{marker}")

lgb_auc_val = results['lgb_baseline']['auc']
best_nf = max([(k, v.get('auc', 0)) for k, v in results.items() if k != 'lgb_baseline'], key=lambda x: x[1])
print(f"\nLightGBM AUC: {lgb_auc_val:.4f}")
print(f"Best NeuralForecast: {best_nf[0]} AUC={best_nf[1]:.4f}")
print(f"Difference: {best_nf[1] - lgb_auc_val:+.4f}")

if best_nf[1] > lgb_auc_val + 0.005:
    print("\n→ NeuralForecast BEATS LightGBM!")
elif best_nf[1] > lgb_auc_val - 0.005:
    print("\n→ NeuralForecast ≈ LightGBM (within noise)")
else:
    print("\n→ LightGBM STILL BETTER — confirms gradient boosting dominance on tabular data")

print(f"\nReference: Phase 4 LGB AUC=0.678 (full 78 features, 35K samples)")
print(f"Reference: ResCNN AUC=0.675 (4 features, 270K samples)")

sys.stdout.flush()

results['config'] = {
    'window': WINDOW, 'horizon': HORIZON, 'seed': SEED,
    'n_tokens': n_tokens, 'n_total': n_total,
}

with open(DATA / 'features/neuralforecast_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)
print(f"\nSaved to {DATA / 'features/neuralforecast_results.json'}")
print(f"Total time: {(time.time()-t0)/60:.1f} min")
sys.stdout.flush()
