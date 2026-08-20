# Prediction Market ML Trading System

> End-to-end ML system for prediction market trading. **521K markets** ingested · **78 engineered features** · **7 model architectures** evaluated · walk-forward validated · deployed live to a VPS with A/B testing.

> **Verdict: NO-GO** — backtest edge collapsed **86×** under live conditions (alpha decay + 2026 fee rollout). Project documents the rigorous research-stop, not a profitable strategy.

## Highlights

- **Walk-forward validated (offline)**: 5/5 windows profitable, permutation test **p = 0.003**, **$8.61/trade** out-of-sample edge
- **LightGBM production model** with test AUC **0.678** — beats every deep-learning architecture on tabular features
- **8-day live deployment** on Ubuntu VPS: 7 strategies in parallel, checkpointing, auto-restart, equity tracking
- **Rigorous validation pipeline**: PurgedKFold (AFML), Platt + isotonic calibration, Monte Carlo stress test, **Deflated Sharpe Ratio** (López de Prado Ch.8)
- **Production-grade engineering**: 117 tests, half-Kelly risk manager, fee-aware backtest, FinBERT sentiment features
- **Identified data leakage** in metadata-only model variant — flagged and rejected
- **Honest validation**: live data revealed an edge collapse — research-stop decision before any real capital was deployed

---

## ML Pipeline

```
Data Collection         3 REST APIs + WebSocket + NLP (Google News RSS, FinBERT)
       ↓
Data Processing         521K markets, 4.2GB raw data, validation, deduplication
       ↓
EDA                     Stationarity testing, whale detection, fat tails, cointegration
       ↓
Feature Engineering     78 features: price/technical, volume, NLP sentiment, cross-market
       ↓
Modeling                LightGBM · XGBoost · CNN 1D · Transformer (LSTM, GRU, DistilBERT also evaluated)
       ↓
Validation              PurgedKFold (AFML), walk-forward (5/5), calibration (Platt, isotonic)
       ↓
Backtesting             Fee-aware simulation, half-Kelly sizing, Monte Carlo stress test
       ↓
Deployment              VPS (Ubuntu), tmux auto-restart, checkpoint/resume, equity logging
       ↓
Paper Trading A/B       7 configs in parallel, 350+ live trades over 8 days
       ↓
Quality & Analysis      Deflated Sharpe Ratio (López de Prado Ch.8), root cause analysis
```

---

## Key Results

| Stage | Metric | Value |
|:------|:-------|------:|
| Classical ML | LightGBM test AUC (TB labels) | **0.678** |
| Classical ML | XGBoost test AUC | 0.677 |
| Deep Learning | ResCNN test AUC | 0.675 |
| Walk-Forward (5 windows) | Mean AUC | 0.683 ± 0.025 |
| Walk-Forward | Profitable windows | **5 / 5** |
| Backtest (HTR cost model) | Flat-bet edge per trade | **$8.61** |
| Backtest | Permutation test p-value | **0.003** |
| Backtest | Win rate / Profit factor | 79% / 2.22 |
| Validation | Identified data leakage in metadata-only model | flagged → rebuild |
| Paper Trading (main, 8 days) | Trades / PnL / WR | **248 / −$25.88 / 47%** |
| Paper Trading | All 7 configs | **all unprofitable** |
| Paper Trading | Deflated Sharpe Ratio | **−0.86** (not significant) |
| Verdict | Phase 7 (live) | **NO-GO** |

---

## Model Comparison

![Model Comparison](assets/model_comparison.png)

LightGBM beats every DL architecture on tabular features. DistilBERT has value only as a stacking feature (uncorrelated text signal), not standalone.

---

## Paper Trading — Case Study (Phase 6)

### The core finding: 86× edge collapse

| Stage | Per-trade edge |
|---|---:|
| Backtest (2024–early-2025 data) | **+$8.61** |
| Paper trading (2026 deployment) | **−$0.10** |
| **Decay factor** | **−86×** |

This is the project's most important number — it quantifies how much regime change between data collection and deployment cost the strategy. Detailed analysis below.

## Hybrid Pipeline: Rule-Based + ML

```
┌──────────────────────────────────────────────────────────┐
│  ML Signal Engine                                        │
│  LightGBM → Platt calibration → calibrated P(outcome)   │
├──────────────────────────────────────────────────────────┤
│  Rule-Based Strategies                                   │
│  Mean Reversion · Contrarian · NegRisk Arb · Convergence │
├──────────────────────────────────────────────────────────┤
│  Meta-Labeling Filter (López de Prado, AFML Ch.3)        │
│  P(primary correct) ≥ 0.6 → trade; else skip            │
├──────────────────────────────────────────────────────────┤
│  Risk Manager                                            │
│  Half-Kelly · Drawdown protection · Regime-aware exits   │
└──────────────────────────────────────────────────────────┘
```

---

## Experiments

| ✅ Applied | ❌ Rejected |
|:---|:---|
| Meta-Labeling — WR 60% → 78% (AFML Ch.3) | Trend-Scanning — MR markets ≠ trending |
| Clustered Feature Importance — 10/78 noise | NeuralForecast — convergence-bias in eval |
| Focal Loss γ=1 — recall 19% → 89% | DL ensemble — inter-model corr 0.93 |
| Deflated Sharpe Ratio — validates NO-GO | Metadata-only HTR (v0) — leakage flagged |

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| **ML** | LightGBM, XGBoost, scikit-learn, Optuna, SHAP |
| **DL** | PyTorch (CNN 1D, ResCNN, Transformer; LSTM/GRU/DistilBERT also evaluated), MPS |
| **NLP** | HuggingFace Transformers, FinBERT (ProsusAI/finbert) for sentiment features |
| **Data** | pandas, numpy, polars, DuckDB, httpx, websockets, asyncio |
| **Feature Engineering** | scipy (ADF, Hurst), statsmodels (cointegration, VR), NMI clustering |
| **Visualization** | matplotlib, seaborn, plotly |
| **Deployment** | VPS (Ubuntu 22.04), tmux, SSH, auto-restart, checkpoint/resume |
| **Validation** | PurgedKFold (AFML), walk-forward, Deflated Sharpe Ratio, Monte Carlo |
| **Risk** | Kelly criterion, half-Kelly, drawdown protection, regime detection |
| **APIs** | REST (httpx), WebSocket (websockets), RSS feeds, JSON/JSONL streaming |
| **Testing** | pytest (117 tests) |
| **Version Control** | Git, GitHub, conda (environment.yml) |

---

**Full list:**

| # | Notebook | Key Result |
|---|----------|------------|
| 1.1 | Market Overview | 50,896 markets, $4.45B volume |
| 1.2 | Resolved Markets | 321K resolved, YES bias +0.217 |
| 1.3 | Price Dynamics | 77% mean-reverting (VR<1), kurtosis ≈ 4140 |
| 1.4 | Strategy Conclusions | Contrarian SR baseline, momentum loses money |
| 1.5 | Trader Analysis | Whale Gini = 0.918, top 1% = 61.5% volume |
| 1.6 | Risk Analysis | Slippage model R² = 0.32 |
| 2.1 | Feature Engineering | 78 features, time-based split |
| 3.1 | Classical ML | LGB AUC = 0.678, Optuna, PurgedKFold |
| 3.2 | Advanced Modeling | Calibration, stacking, walk-forward |
| 4.1 | Backtesting | Fee-aware HTR cost model |
| 4.2 | Model Improvement | LGB v3 + volume features, calibration |
| 4.3 | Validation | Walk-forward 5/5 profitable, p=0.003; identified leakage in v0 |
| 4.4 | Paper Trading Analysis | 7 A/B configs, NO-GO verdict, root cause |
| 5.1 | CNN 1D Time Series | ResCNN AUC = 0.675 |
| 5.2 | Transformer Time Series | CLS Transformer AUC = 0.670 |
| 6.1 | Clustered Feature Importance | NMI + ONC, 10/78 features = noise |

</details>

---
