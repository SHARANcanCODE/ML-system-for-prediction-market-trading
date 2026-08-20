# Data Guide

## Overview

This project uses data from [Polymarket](https://polymarket.com) — a prediction market platform. All read-only API endpoints are free and require no authentication.

For full API documentation, see [docs.polymarket.com](https://docs.polymarket.com).

## Data Sources

| Source | Data | Used For |
|--------|------|----------|
| Gamma API | Market metadata, categories, volumes | EDA, filtering |
| CLOB API | Price history, midpoints, orderbooks | Features, backtesting |
| Data API | Trade history per market | Volume analysis, whale detection |
| Google News RSS | News headlines by keywords | NLP sentiment features |

## Quick Start

### Option 1: Sample dataset (recommended for testing)

A small sample dataset (10 markets, 7 days) is included in `data/sample/`:

```bash
# Notebooks automatically detect and use sample data if full dataset is missing
jupyter lab notebooks/
```

### Option 2: Full dataset collection

```bash
# Collect top 500 markets with 90 days of price history
python scripts/collect_data.py --limit 500 --prices-days 90

# Build feature matrix from collected data
python -m src.data.pipeline
```

Collection takes ~30 minutes for 500 markets. Progress is saved incrementally — safe to interrupt and resume.

## Data Structure

```
data/
├── raw/                    # Raw API responses (JSONL)
│   ├── markets/            # Market metadata snapshots
│   ├── prices/             # Price history (hourly bars)
│   └── trades/             # Trade records
├── processed/              # Cleaned and merged datasets
│   ├── markets.parquet     # Market catalog
│   ├── prices.parquet      # OHLCV-style price bars
│   └── trades.parquet      # Normalized trades
├── features/               # Computed features
│   └── feature_matrix.parquet  # 78-column feature matrix
├── models/                 # Trained model artifacts
│   ├── lgb_true_htr_v1.joblib  # Production HTR model
│   ├── calibrators_v3.joblib   # Probability calibrators
│   └── meta_minimal_v1.joblib  # Meta-labeling model
└── sample/                 # Small demo dataset
```

## Pre-trained Models

Models in `data/models/` are trained on the full dataset (~500 markets, 90 days). They are included for inference and analysis — retraining requires the full dataset.
