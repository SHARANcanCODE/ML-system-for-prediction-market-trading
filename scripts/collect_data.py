from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, ".")

from src.data.client import PolymarketClient
from src.utils.logger import get_logger

log = get_logger("collect_all")

RAW_DIR = Path("data/raw")
PROGRESS_DIR = Path("data/raw/.progress")

shutdown_event = threading.Event()

def _signal_handler(sig, frame):
    log.info("Shutdown signal received — finishing current items and saving...")
    shutdown_event.set()

signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

progress_lock = threading.Lock()
progress = {}

def _init_progress(name: str, total: int, skipped: int = 0):
    with progress_lock:
        progress[name] = {
            "done": skipped,
            "total": total,
            "skipped": skipped,
            "items": 0,
            "errors": 0,
            "started_at": time.time(),
        }

def _update_progress(name: str, items: int = 0, error: bool = False):
    with progress_lock:
        p = progress[name]
        p["done"] += 1
        p["items"] += items
        if error:
            p["errors"] += 1

def _get_progress_summary() -> str:
    with progress_lock:
        parts = []
        for name, p in sorted(progress.items()):
            done = p["done"]
            total = p["total"]
            pct = done / total * 100 if total else 0
            elapsed = time.time() - p["started_at"]
            active = done - p["skipped"]
            rate = active / elapsed * 60 if elapsed > 0 and active > 0 else 0
            remaining = total - done
            eta_min = remaining / (active / elapsed * 60) if active > 0 and elapsed > 0 else 0

            bar_len = 15
            filled = int(bar_len * done / total) if total else 0
            bar = "=" * filled + ">" + "." * (bar_len - filled - 1) if filled < bar_len else "=" * bar_len

            parts.append(
                f"{name}: [{bar}] {done}/{total} ({pct:.0f}%) "
                f"| {p['items']:,} items | {rate:.1f}/min | ETA {eta_min:.0f}m"
                + (f" | {p['errors']} err" if p["errors"] else "")
            )
        return "\n".join(parts)

def _save_progress_file():
    summary = _get_progress_summary()
    path = PROGRESS_DIR / "STATUS.txt"
    elapsed = time.time() - min(
        (p["started_at"] for p in progress.values()), default=time.time()
    )
    header = (
        f"Collection Status — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
        f"Elapsed: {elapsed/60:.1f} min\n"
        f"{'=' * 70}\n"
    )
    path.write_text(header + summary + "\n")

def progress_reporter():
    while not shutdown_event.is_set():
        shutdown_event.wait(30)
        if shutdown_event.is_set():
            break
        if progress:
            log.info(f"\n{'─' * 60}\n{_get_progress_summary()}\n{'─' * 60}")
            _save_progress_file()

    if progress:
        _save_progress_file()

def _load_progress(name: str) -> set[str]:
    PROGRESS_DIR.mkdir(parents=True, exist_ok=True)
    path = PROGRESS_DIR / f"{name}.done"
    if path.exists():
        return set(path.read_text().strip().split("\n"))
    return set()

def _mark_done(name: str, item_id: str):
    path = PROGRESS_DIR / f"{name}.done"
    with open(path, "a") as f:
        f.write(item_id + "\n")

def _append_jsonl(path: Path, data: dict | list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        if isinstance(data, list):
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        else:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

def load_markets(limit: int) -> list[dict]:
    markets_file = "data/processed/top_markets_for_collection.json"
    with open(markets_file) as f:
        markets = json.load(f)
    return markets[:limit]

def extract_tokens(markets: list[dict]) -> list[tuple[str, str, str]]:
    result = []
    for m in markets:
        cid = m.get("conditionId", "")
        question = m.get("question", "")[:60]
        raw = m.get("clobTokenIds", "[]")
        try:
            ids = json.loads(raw) if isinstance(raw, str) else raw
        except json.JSONDecodeError:
            ids = []
        for tid in ids:
            if tid:
                result.append((tid, cid, question))
    return result

def stream_trades(markets: list[dict], max_trades: int = 10000, delay: float = 0.12):
    stream_name = "trades"
    done = _load_progress(stream_name)
    out_path = RAW_DIR / "trades" / f"trades_incremental_{datetime.now(timezone.utc).strftime('%Y%m%d')}.jsonl"

    total = len(markets)
    skipped = sum(1 for m in markets if m.get("conditionId", "") in done)
    _init_progress(stream_name, total, skipped)

    collected = 0
    client = PolymarketClient()

    try:
        for i, market in enumerate(markets):
            if shutdown_event.is_set():
                break

            cid = market.get("conditionId", "")
            if not cid or cid in done:
                continue

            all_trades = []
            batch_size = 100
            had_error = False
            while len(all_trades) < max_trades:
                if shutdown_event.is_set():
                    break
                try:
                    trades = client.get_trades(
                        market=cid, limit=batch_size, offset=len(all_trades)
                    )
                except Exception as e:
                    log.warning(f"[trades] {cid[:16]}... failed: {e}")
                    had_error = True
                    time.sleep(1)
                    break

                if not trades:
                    break
                all_trades.extend(trades)
                if len(trades) < batch_size:
                    break
                time.sleep(delay)

            if all_trades:
                _append_jsonl(out_path, all_trades)
                collected += len(all_trades)

            _mark_done(stream_name, cid)
            _update_progress(stream_name, items=len(all_trades), error=had_error)
            time.sleep(delay)

    finally:
        client.close()

    log.info(f"[trades] DONE: {collected} trades from {total - skipped} markets -> {out_path}")

def stream_prices(
    tokens: list[tuple[str, str, str]],
    fidelity: int = 60,
    days: int = 90,
    delay: float = 0.12,
    clob_lock: threading.Lock | None = None,
):
    stream_name = f"prices_f{fidelity}"
    done = _load_progress(stream_name)
    out_path = (
        RAW_DIR
        / "prices"
        / f"prices_f{fidelity}_incremental_{datetime.now(timezone.utc).strftime('%Y%m%d')}.jsonl"
    )

    now = int(time.time())
    start_ts = now - days * 24 * 3600
    max_window = 7 * 24 * 3600

    total = len(tokens)
    skipped = sum(1 for t in tokens if t[0] in done)
    _init_progress(stream_name, total, skipped)

    collected = 0
    client = PolymarketClient()

    try:
        for i, (token_id, cid, question) in enumerate(tokens):
            if shutdown_event.is_set():
                break

            if token_id in done:
                continue

            all_history = []
            current_start = start_ts
            had_error = False
            while current_start < now:
                if shutdown_event.is_set():
                    break
                current_end = min(current_start + max_window, now)
                try:
                    if clob_lock:
                        clob_lock.acquire()
                    try:
                        history = client.get_price_history(
                            token_id=token_id,
                            start_ts=current_start,
                            end_ts=current_end,
                            fidelity=fidelity,
                        )
                    finally:
                        if clob_lock:
                            clob_lock.release()
                    all_history.extend(history)
                except Exception as e:
                    log.warning(f"[prices f{fidelity}] {token_id[:16]}... failed: {e}")
                    had_error = True
                    time.sleep(1)

                current_start = current_end
                time.sleep(delay)

            if all_history:
                record = {
                    "token_id": token_id,
                    "condition_id": cid,
                    "fidelity": fidelity,
                    "n_bars": len(all_history),
                    "history": all_history,
                }
                _append_jsonl(out_path, record)
                collected += len(all_history)

            _mark_done(stream_name, token_id)
            _update_progress(stream_name, items=len(all_history), error=had_error)

    finally:
        client.close()

    log.info(f"[prices f{fidelity}] DONE: {collected} bars from {total - skipped} tokens -> {out_path}")

def stream_orderbooks(
    tokens: list[tuple[str, str, str]],
    delay: float = 0.1,
    clob_lock: threading.Lock | None = None,
):
    stream_name = "orderbooks"
    done = _load_progress(stream_name)
    ts_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = RAW_DIR / "orderbooks" / f"orderbooks_full_{ts_str}.jsonl"

    total = len(tokens)
    skipped = sum(1 for t in tokens if t[0] in done)
    _init_progress(stream_name, total, skipped)

    collected = 0
    client = PolymarketClient()

    try:
        for i, (token_id, cid, question) in enumerate(tokens):
            if shutdown_event.is_set():
                break

            if token_id in done:
                continue

            had_error = False
            try:
                if clob_lock:
                    clob_lock.acquire()
                try:
                    book = client.get_order_book(token_id)
                finally:
                    if clob_lock:
                        clob_lock.release()

                if hasattr(book, "__dict__"):
                    book = book.__dict__

                bids = book.get("bids", [])
                asks = book.get("asks", [])

                record = {
                    "ts": time.time(),
                    "token_id": token_id,
                    "condition_id": cid,
                    "best_bid": float(bids[0]["price"]) if bids else None,
                    "best_ask": float(asks[0]["price"]) if asks else None,
                    "bid_depth_5": sum(float(b.get("size", 0)) for b in bids[:5]),
                    "ask_depth_5": sum(float(a.get("size", 0)) for a in asks[:5]),
                    "bid_depth_all": sum(float(b.get("size", 0)) for b in bids),
                    "ask_depth_all": sum(float(a.get("size", 0)) for a in asks),
                    "spread": (float(asks[0]["price"]) - float(bids[0]["price"]))
                    if bids and asks
                    else None,
                    "n_bid_levels": len(bids),
                    "n_ask_levels": len(asks),
                }
                _append_jsonl(out_path, record)
                collected += 1
            except Exception as e:
                log.debug(f"[orderbooks] {token_id[:16]}... failed: {e}")
                had_error = True

            _mark_done(stream_name, token_id)
            _update_progress(stream_name, items=1 if not had_error else 0, error=had_error)
            time.sleep(delay)

    finally:
        client.close()

    log.info(f"[orderbooks] DONE: {collected} snapshots -> {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Parallel Polymarket data collection")
    parser.add_argument("--limit", type=int, default=50, help="Number of top markets (default: 50)")
    parser.add_argument(
        "--streams",
        nargs="+",
        choices=["trades", "prices", "prices5", "orderbooks"],
        default=None,
        help="Which streams to run (default: all). prices=60min, prices5=5min bars",
    )
    parser.add_argument("--prices-fidelity", type=int, default=60, help="Price bar fidelity in minutes")
    parser.add_argument("--prices-days", type=int, default=90, help="How many days of price history")
    parser.add_argument("--max-trades", type=int, default=10000, help="Max trades per market")
    parser.add_argument("--delay", type=float, default=0.12, help="Delay between API calls (seconds)")
    parser.add_argument("--reset", action="store_true", help="Clear progress files and start fresh")
    args = parser.parse_args()

    streams = args.streams or ["trades", "prices", "orderbooks"]

    if args.reset:
        if PROGRESS_DIR.exists():
            for f in PROGRESS_DIR.glob("*.done"):
                f.unlink()
            log.info("Progress files cleared")

    markets = load_markets(args.limit)
    tokens = extract_tokens(markets)
    log.info(f"Loaded {len(markets)} markets, {len(tokens)} tokens")

    for stream in streams:
        name = stream if stream != "prices5" else "prices_f5"
        if stream == "prices":
            name = f"prices_f{args.prices_fidelity}"
        done = _load_progress(name)
        if done:
            log.info(f"  [{name}] {len(done)} already collected, will skip")

    clob_lock = threading.Lock()

    threads = []

    if "trades" in streams:
        t = threading.Thread(
            target=stream_trades,
            args=(markets, args.max_trades, args.delay),
            name="trades",
        )
        threads.append(t)

    if "prices" in streams:
        t = threading.Thread(
            target=stream_prices,
            args=(tokens, args.prices_fidelity, args.prices_days, args.delay, clob_lock),
            name="prices",
        )
        threads.append(t)

    if "prices5" in streams:
        t = threading.Thread(
            target=stream_prices,
            args=(tokens, 5, min(args.prices_days, 30), args.delay, clob_lock),
            name="prices5",
        )
        threads.append(t)

    if "orderbooks" in streams:
        t = threading.Thread(
            target=stream_orderbooks,
            args=(tokens, args.delay, clob_lock),
            name="orderbooks",
        )
        threads.append(t)

    log.info(f"Starting {len(threads)} parallel streams: {[t.name for t in threads]}")
    log.info(f"Monitor progress: cat data/raw/.progress/STATUS.txt")
    start_time = time.time()

    reporter = threading.Thread(target=progress_reporter, name="reporter", daemon=True)
    reporter.start()

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    shutdown_event.set()
    reporter.join(timeout=2)

    elapsed = time.time() - start_time
    minutes = elapsed / 60

    if progress:
        log.info(f"\n{'=' * 60}\nFINAL REPORT ({minutes:.1f} min)\n{'=' * 60}\n{_get_progress_summary()}\n{'=' * 60}")
        _save_progress_file()

    if any(p["done"] < p["total"] for p in progress.values()):
        log.info(f"Collection interrupted after {minutes:.1f} min. Run again to resume.")
    else:
        log.info(f"All streams completed in {minutes:.1f} min.")

    log.info("Run pipeline to rebuild processed data: python scripts/update_data.py --pipeline-only")

if __name__ == "__main__":
    main()
