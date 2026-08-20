import argparse
import json
import signal
import time
from datetime import datetime, timezone
from pathlib import Path

from src.data.client import PolymarketClient
from src.utils.logger import get_logger

log = get_logger(__name__)

RAW_DIR = Path("data/raw")
SNAPSHOTS_DIR = RAW_DIR / "snapshots"

def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

def _date_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")

def _append_jsonl(path: Path, record: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def load_markets(file_path: str) -> list[dict]:
    with open(file_path) as f:
        return json.load(f)

def _parse_token_ids(markets: list[dict]) -> list[tuple[str, str]]:
    pairs = []
    for m in markets:
        cid = m.get("conditionId", "")
        raw = m.get("clobTokenIds", "[]")
        token_ids = json.loads(raw) if isinstance(raw, str) else raw
        for tid in token_ids:
            if tid:
                pairs.append((tid, cid))
    return pairs

def snapshot_orderbooks(
    client: PolymarketClient,
    token_pairs: list[tuple[str, str]],
    delay: float = 0.05,
) -> int:
    out_path = SNAPSHOTS_DIR / f"orderbooks_{_date_str()}.jsonl"
    ts = time.time()
    count = 0

    for token_id, cid in token_pairs:
        try:
            book = client.get_order_book(token_id)

            if hasattr(book, "__dict__"):
                book = book.__dict__

            bids = book.get("bids", [])
            asks = book.get("asks", [])

            record = {
                "ts": ts,
                "token_id": token_id,
                "condition_id": cid,
                "best_bid": float(bids[0]["price"]) if bids else None,
                "best_ask": float(asks[0]["price"]) if asks else None,
                "bid_depth": sum(float(b.get("size", 0)) for b in bids[:5]),
                "ask_depth": sum(float(a.get("size", 0)) for a in asks[:5]),
                "spread": (float(asks[0]["price"]) - float(bids[0]["price"]))
                if bids and asks
                else None,
                "n_bid_levels": len(bids),
                "n_ask_levels": len(asks),
            }
            _append_jsonl(out_path, record)
            count += 1
        except Exception as e:
            log.debug(f"Orderbook failed {token_id[:12]}...: {e}")

        time.sleep(delay)

    log.info(f"Orderbook snapshot: {count}/{len(token_pairs)} tokens → {out_path.name}")
    return count

def snapshot_prices(
    client: PolymarketClient,
    token_pairs: list[tuple[str, str]],
    delay: float = 0.05,
) -> int:
    out_path = SNAPSHOTS_DIR / f"prices_{_date_str()}.jsonl"
    ts = time.time()
    count = 0

    for token_id, cid in token_pairs:
        try:
            mid = client.get_midpoint(token_id)
            record = {
                "ts": ts,
                "token_id": token_id,
                "condition_id": cid,
                "midpoint": mid,
            }
            _append_jsonl(out_path, record)
            count += 1
        except Exception as e:
            log.debug(f"Price failed {token_id[:12]}...: {e}")

        time.sleep(delay)

    log.info(f"Price snapshot: {count}/{len(token_pairs)} tokens → {out_path.name}")
    return count

def monitor_new_markets(
    client: PolymarketClient,
    known_ids: set[str],
) -> list[dict]:
    out_path = SNAPSHOTS_DIR / f"new_markets_{_date_str()}.jsonl"
    new_markets = []

    try:

        events = client.get_events(active=True, closed=False, limit=100, offset=0)

        for event in events:
            for market in event.get("markets", []):
                mid = market.get("conditionId", "")
                if mid and mid not in known_ids:
                    known_ids.add(mid)
                    market["_event_title"] = event.get("title", "")
                    market["_detected_ts"] = time.time()
                    new_markets.append(market)
                    _append_jsonl(out_path, market)

    except Exception as e:
        log.warning(f"New market check failed: {e}")

    if new_markets:
        log.info(f"Detected {len(new_markets)} new markets")
    return new_markets

def run_scheduler(
    markets_file: str = "data/processed/top_markets_for_collection.json",
    interval_minutes: int = 15,
    duration_minutes: int = 1440,
    jobs: list[str] | None = None,
):
    if jobs is None:
        jobs = ["orderbooks", "prices", "new_markets"]

    markets = load_markets(markets_file)
    token_pairs = _parse_token_ids(markets)
    log.info(f"Loaded {len(markets)} markets, {len(token_pairs)} tokens")

    known_ids = {m.get("conditionId", "") for m in markets}

    client = PolymarketClient()
    interval_sec = interval_minutes * 60
    end_time = time.time() + duration_minutes * 60
    cycle = 0

    shutdown = False

    def _signal_handler(sig, frame):
        nonlocal shutdown
        log.info("Shutdown signal received, finishing current cycle...")
        shutdown = True

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    log.info(
        f"Scheduler started: interval={interval_minutes}min, "
        f"duration={duration_minutes}min, jobs={jobs}"
    )

    try:
        while time.time() < end_time and not shutdown:
            cycle += 1
            cycle_start = time.time()
            log.info(f"--- Cycle {cycle} ---")

            if "orderbooks" in jobs:
                snapshot_orderbooks(client, token_pairs)

            if "prices" in jobs:
                snapshot_prices(client, token_pairs)

            if "new_markets" in jobs:
                new = monitor_new_markets(client, known_ids)

                for m in new:
                    vol = float(m.get("volume", 0) or 0)
                    if vol > 10_000:
                        raw = m.get("clobTokenIds", "[]")
                        tids = json.loads(raw) if isinstance(raw, str) else raw
                        cid = m.get("conditionId", "")
                        for tid in tids:
                            if tid and (tid, cid) not in token_pairs:
                                token_pairs.append((tid, cid))
                        log.info(f"Added new market to tracking: {m.get('question', '')[:60]}")

            elapsed = time.time() - cycle_start
            sleep_time = max(0, interval_sec - elapsed)
            log.info(f"Cycle {cycle} done in {elapsed:.1f}s, sleeping {sleep_time:.0f}s")

            if sleep_time > 0 and not shutdown:

                chunks = int(sleep_time / 5) + 1
                for _ in range(chunks):
                    if shutdown:
                        break
                    time.sleep(min(5, sleep_time))
                    sleep_time -= 5

    finally:
        client.close()
        log.info(f"Scheduler stopped after {cycle} cycles")

def main():
    parser = argparse.ArgumentParser(description="Polymarket periodic data collector")
    parser.add_argument(
        "--markets-file",
        default="data/processed/top_markets_for_collection.json",
        help="Path to JSON with market list",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=15,
        help="Minutes between snapshots (default: 15)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=1440,
        help="Total runtime in minutes (default: 1440 = 24h)",
    )
    parser.add_argument(
        "--jobs",
        nargs="+",
        choices=["orderbooks", "prices", "new_markets"],
        default=None,
        help="Which jobs to run (default: all)",
    )
    args = parser.parse_args()

    run_scheduler(
        markets_file=args.markets_file,
        interval_minutes=args.interval,
        duration_minutes=args.duration,
        jobs=args.jobs,
    )

if __name__ == "__main__":
    main()
