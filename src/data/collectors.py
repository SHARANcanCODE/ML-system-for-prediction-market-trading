import json
import time
from datetime import datetime, timezone
from pathlib import Path

from src.data.client import PolymarketClient
from src.utils.logger import get_logger

log = get_logger(__name__)

RAW_DIR = Path("data/raw")

def _timestamp_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

def _save_json(data: list | dict, name: str, subdir: str = "") -> Path:
    target = RAW_DIR / subdir if subdir else RAW_DIR
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"{name}_{_timestamp_str()}.json"
    with open(path, "w") as f:
        json.dump(data, f, ensure_ascii=False)
    log.info(f"Saved {path} ({len(data) if isinstance(data, list) else 1} records)")
    return path

def collect_all_events(
    client: PolymarketClient,
    active: bool = True,
    closed: bool = False,
    batch_size: int = 100,
    delay: float = 0.1,
) -> list[dict]:
    all_events = []
    offset = 0

    while True:
        events = client.get_events(
            active=active,
            closed=closed,
            limit=batch_size,
            offset=offset,
        )

        if not events:
            break

        all_events.extend(events)
        log.info(f"Fetched {len(all_events)} events (offset={offset})")

        if len(events) < batch_size:
            break

        offset += batch_size
        time.sleep(delay)

    return all_events

def collect_markets(
    client: PolymarketClient,
    include_active: bool = True,
    include_resolved: bool = True,
    save: bool = True,
) -> dict[str, list[dict]]:
    all_events = []

    if include_active:
        log.info("Collecting active events...")
        active = collect_all_events(client, active=True, closed=False)
        all_events.extend(active)
        log.info(f"Active events: {len(active)}")

    if include_resolved:
        log.info("Collecting resolved events...")
        resolved = collect_all_events(client, active=False, closed=True)
        all_events.extend(resolved)
        log.info(f"Resolved events: {len(resolved)}")

    markets = []
    for event in all_events:
        event_id = event.get("id", "")
        event_slug = event.get("slug", "")
        event_title = event.get("title", "")
        neg_risk = event.get("negRisk", False)
        volume_24hr = event.get("volume24hr", 0)

        for market in event.get("markets", []):
            market["_event_id"] = event_id
            market["_event_slug"] = event_slug
            market["_event_title"] = event_title
            market["_event_neg_risk"] = neg_risk
            market["_event_volume_24hr"] = volume_24hr
            markets.append(market)

    log.info(f"Total: {len(all_events)} events, {len(markets)} markets")

    result = {"events": all_events, "markets": markets}

    if save:
        _save_json(all_events, "events", subdir="markets")
        _save_json(markets, "markets_flat", subdir="markets")

    return result

def collect_price_history(
    client: PolymarketClient,
    token_id: str,
    start_ts: int,
    end_ts: int,
    fidelity: int = 60,
    max_window: int = 7 * 24 * 3600,
    delay: float = 0.1,
) -> list[dict]:
    all_history = []
    current_start = start_ts

    while current_start < end_ts:
        current_end = min(current_start + max_window, end_ts)

        try:
            history = client.get_price_history(
                token_id=token_id,
                start_ts=current_start,
                end_ts=current_end,
                fidelity=fidelity,
            )
            all_history.extend(history)
        except Exception as e:
            log.warning(f"Price history failed for {token_id} [{current_start}-{current_end}]: {e}")

        current_start = current_end
        if current_start < end_ts:
            time.sleep(delay)

    return all_history

def collect_prices_for_markets(
    client: PolymarketClient,
    markets: list[dict],
    start_ts: int,
    end_ts: int,
    fidelity: int = 60,
    save: bool = True,
    delay: float = 0.15,
) -> dict[str, list[dict]]:
    all_prices = {}
    total = len(markets)

    for i, market in enumerate(markets):
        clob_ids_raw = market.get("clobTokenIds", "[]")
        try:
            token_ids = json.loads(clob_ids_raw) if isinstance(clob_ids_raw, str) else clob_ids_raw
        except json.JSONDecodeError:
            continue

        for token_id in token_ids:
            if not token_id:
                continue
            history = collect_price_history(
                client, token_id, start_ts, end_ts, fidelity, delay=delay
            )
            if history:
                all_prices[token_id] = history

        if (i + 1) % 50 == 0:
            log.info(f"Price history: {i + 1}/{total} markets processed")

        time.sleep(delay)

    log.info(f"Collected prices for {len(all_prices)} tokens from {total} markets")

    if save and all_prices:
        _save_json(all_prices, f"prices_f{fidelity}", subdir="prices")

    return all_prices

def collect_trades(
    client: PolymarketClient,
    condition_id: str,
    max_trades: int = 10000,
    delay: float = 0.1,
) -> list[dict]:
    all_trades = []
    batch_size = 100

    while len(all_trades) < max_trades:
        try:
            trades = client.get_trades(
                market=condition_id,
                limit=batch_size,
                offset=len(all_trades),
            )
        except Exception as e:
            log.warning(f"Trades fetch failed for {condition_id}: {e}")
            break

        if not trades:
            break

        all_trades.extend(trades)

        if len(trades) < batch_size:
            break

        time.sleep(delay)

    return all_trades

def collect_trades_for_markets(
    client: PolymarketClient,
    markets: list[dict],
    save: bool = True,
    max_trades_per_market: int = 10000,
    delay: float = 0.15,
) -> dict[str, list[dict]]:
    all_trades = {}
    total = len(markets)

    for i, market in enumerate(markets):
        cid = market.get("conditionId", "")
        if not cid:
            continue

        trades = collect_trades(client, cid, max_trades=max_trades_per_market, delay=delay)
        if trades:
            all_trades[cid] = trades

        if (i + 1) % 50 == 0:
            log.info(f"Trades: {i + 1}/{total} markets, {sum(len(t) for t in all_trades.values())} total trades")

        time.sleep(delay)

    log.info(f"Collected {sum(len(t) for t in all_trades.values())} trades from {len(all_trades)} markets")

    if save and all_trades:
        _save_json(all_trades, "trades", subdir="trades")

    return all_trades

def collect_order_book_snapshots(
    client: PolymarketClient,
    markets: list[dict],
    save: bool = True,
    delay: float = 0.1,
) -> dict[str, dict]:
    snapshots = {}
    total = len(markets)

    for i, market in enumerate(markets):
        clob_ids_raw = market.get("clobTokenIds", "[]")
        try:
            token_ids = json.loads(clob_ids_raw) if isinstance(clob_ids_raw, str) else clob_ids_raw
        except json.JSONDecodeError:
            continue

        for token_id in token_ids:
            if not token_id:
                continue
            try:
                book = client.get_order_book(token_id)

                if hasattr(book, "__dict__"):
                    book = book.__dict__
                snapshots[token_id] = book
            except Exception as e:
                log.warning(f"Order book failed for {token_id}: {e}")

            time.sleep(delay)

        if (i + 1) % 100 == 0:
            log.info(f"Order books: {i + 1}/{total} markets, {len(snapshots)} snapshots")

    log.info(f"Collected {len(snapshots)} order book snapshots")

    if save and snapshots:
        _save_json(snapshots, "orderbooks", subdir="orderbooks")

    return snapshots
