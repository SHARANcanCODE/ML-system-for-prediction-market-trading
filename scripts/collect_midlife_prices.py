import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.client import PolymarketClient
from src.data.collectors import collect_price_history
from src.utils.logger import get_logger

log = get_logger(__name__)

PROJECT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT / "data"
OUTPUT_FILE = DATA_DIR / "processed" / "midlife_prices.parquet"
CHECKPOINT_FILE = DATA_DIR / "processed" / "midlife_prices_checkpoint.json"

def load_resolved_markets(min_volume: float = 50_000, min_lifetime_h: float = 24) -> pd.DataFrame:
    df = pd.read_parquet(DATA_DIR / "processed" / "resolution_outcomes.parquet")
    df = df.dropna(subset=["createdAt", "closedTime", "token_id_yes"])
    df["lifetime_h"] = (df["closedTime"] - df["createdAt"]).dt.total_seconds() / 3600

    mask = (df["volumeNum"] >= min_volume) & (df["lifetime_h"] >= min_lifetime_h)
    filtered = df[mask].copy()
    log.info(f"Filtered: {len(filtered)} markets (vol>=${min_volume:,.0f}, lifetime>={min_lifetime_h}h)")
    return filtered

def get_midlife_price(
    client: PolymarketClient,
    token_id: str,
    created_ts: int,
    closed_ts: int,
) -> dict | None:
    lifetime = closed_ts - created_ts
    if lifetime < 3600:
        return None

    mid_ts = created_ts + lifetime // 2
    q1_ts = created_ts + lifetime // 4
    q3_ts = created_ts + 3 * lifetime // 4

    fidelity = 60
    window = 6 * 3600

    all_history = []
    targets = [
        ("first", created_ts, min(created_ts + window * 2, closed_ts)),
        ("q1", max(created_ts, q1_ts - window), min(closed_ts, q1_ts + window)),
        ("mid", max(created_ts, mid_ts - window), min(closed_ts, mid_ts + window)),
        ("q3", max(created_ts, q3_ts - window), min(closed_ts, q3_ts + window)),
        ("last", max(created_ts, closed_ts - window * 2), closed_ts),
    ]

    for label, w_start, w_end in targets:
        try:
            history = client.get_price_history(
                token_id=token_id, start_ts=w_start, end_ts=w_end, fidelity=fidelity,
            )
            if history:
                all_history.extend(history)
        except Exception as e:
            log.debug(f"API error for {token_id[:20]} ({label}): {e}")

    if not all_history:
        return None

    seen = set()
    unique = []
    for h in all_history:
        if h["t"] not in seen:
            seen.add(h["t"])
            unique.append(h)
    unique.sort(key=lambda h: h["t"])

    if len(unique) < 2:
        return None

    timestamps = np.array([int(h["t"]) for h in unique])
    all_prices = np.array([float(h["p"]) for h in unique])

    def closest_price(target_ts):
        idx = np.argmin(np.abs(timestamps - target_ts))
        return float(all_prices[idx])

    result = {
        "price_q1": closest_price(q1_ts),
        "price_mid": closest_price(mid_ts),
        "price_q3": closest_price(q3_ts),
        "price_first": float(all_prices[0]),
        "price_last": float(all_prices[-1]),
        "n_bars": len(all_prices),
        "price_std": float(np.std(all_prices)),
        "price_range": float(np.max(all_prices) - np.min(all_prices)),
        "price_mean": float(np.mean(all_prices)),
    }

    result["momentum_first_half"] = result["price_mid"] - result["price_q1"]
    result["momentum_second_half"] = result["price_q3"] - result["price_mid"]

    return result

def collect_midlife_prices(
    markets: pd.DataFrame,
    batch_size: int = 500,
    delay: float = 0.12,
    already_done: set | None = None,
) -> pd.DataFrame:
    client = PolymarketClient()
    already_done = already_done or set()

    results = []
    to_process = markets[~markets["conditionId"].isin(already_done)]
    total = len(to_process)
    log.info(f"Processing {total} markets ({len(already_done)} already done)")

    errors = 0
    batch_results = []

    for i, (_, row) in enumerate(to_process.iterrows()):
        token_id = row["token_id_yes"]
        created_ts = int(row["createdAt"].timestamp())
        closed_ts = int(row["closedTime"].timestamp())

        price_data = get_midlife_price(client, token_id, created_ts, closed_ts)

        if price_data is not None:
            price_data["conditionId"] = row["conditionId"]
            price_data["token_id_yes"] = token_id
            batch_results.append(price_data)
        else:
            errors += 1

        if (i + 1) % 100 == 0:
            success = len(batch_results) + len(results)
            log.info(f"[{i+1}/{total}] Success: {success}, Errors: {errors}")

        if (i + 1) % batch_size == 0 and batch_results:
            results.extend(batch_results)
            _save_checkpoint(results, already_done | {r["conditionId"] for r in results})
            batch_results = []

        time.sleep(delay)

    results.extend(batch_results)
    if results:
        df_results = pd.DataFrame(results)
        _save_results(df_results, already_done)
        log.info(f"Done! {len(results)} new prices collected, {errors} errors")
        return df_results
    else:
        log.warning("No results collected")
        return pd.DataFrame()

def _save_checkpoint(results: list[dict], done_ids: set):
    checkpoint = {
        "n_collected": len(results),
        "done_ids": list(done_ids),
        "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint, f)

    df = pd.DataFrame(results)
    df.to_parquet(OUTPUT_FILE, index=False)
    log.info(f"Checkpoint saved: {len(results)} records")

def _save_results(df: pd.DataFrame, already_done: set):
    if OUTPUT_FILE.exists() and already_done:
        existing = pd.read_parquet(OUTPUT_FILE)
        df = pd.concat([existing, df], ignore_index=True)
        df = df.drop_duplicates(subset=["conditionId"], keep="last")

    df.to_parquet(OUTPUT_FILE, index=False)
    log.info(f"Saved {len(df)} records to {OUTPUT_FILE}")

def load_checkpoint() -> set:
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            data = json.load(f)
        done_ids = set(data.get("done_ids", []))
        log.info(f"Resuming from checkpoint: {len(done_ids)} already done")
        return done_ids
    return set()

def main():
    parser = argparse.ArgumentParser(description="Collect mid-life prices for resolved markets")
    parser.add_argument("--min-volume", type=float, default=50_000,
                        help="Minimum market volume ($)")
    parser.add_argument("--min-lifetime", type=float, default=24,
                        help="Minimum market lifetime (hours)")
    parser.add_argument("--batch-size", type=int, default=500,
                        help="Checkpoint every N markets")
    parser.add_argument("--delay", type=float, default=0.12,
                        help="Delay between API calls (seconds)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit number of markets (0=all)")
    args = parser.parse_args()

    markets = load_resolved_markets(args.min_volume, args.min_lifetime)

    if args.limit > 0:
        markets = markets.head(args.limit)
        log.info(f"Limited to {args.limit} markets")

    already_done = load_checkpoint() if args.resume else set()

    collect_midlife_prices(
        markets=markets,
        batch_size=args.batch_size,
        delay=args.delay,
        already_done=already_done,
    )

if __name__ == "__main__":
    main()
