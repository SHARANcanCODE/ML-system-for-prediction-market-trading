import argparse
import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import websockets

from src.utils.logger import get_logger

log = get_logger(__name__)

WSS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
MAX_TOKENS_PER_CONNECTION = 450
PING_INTERVAL = 30
RECONNECT_DELAY = 5
DATA_DIR = Path("data/raw/websocket")

def _output_path(prefix: str) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d")
    return DATA_DIR / f"{prefix}_{ts}.jsonl"

def _append_jsonl(path: Path, record: dict):
    with open(path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def load_token_ids(file_path: str, max_tokens: int = 50) -> list[str]:
    with open(file_path) as f:
        markets = json.load(f)

    token_ids = []
    for m in markets:
        raw = m.get("clobTokenIds", "[]")
        ids = json.loads(raw) if isinstance(raw, str) else raw
        token_ids.extend(ids)
        if len(token_ids) >= max_tokens:
            break

    token_ids = token_ids[:max_tokens]
    log.info(f"Loaded {len(token_ids)} token IDs from {len(markets)} markets")
    return token_ids

async def collect_stream(
    token_ids: list[str],
    duration_seconds: int = 3600,
    output_prefix: str = "ws_stream",
):
    out_path = _output_path(output_prefix)
    end_time = time.time() + duration_seconds
    event_count = 0
    reconnect_count = 0

    while time.time() < end_time:
        try:
            async with websockets.connect(
                WSS_URL,
                ping_interval=PING_INTERVAL,
                ping_timeout=10,
                close_timeout=5,
            ) as ws:

                sub_msg = json.dumps({
                    "assets_ids": token_ids,
                    "type": "market",
                    "custom_feature_enabled": True,
                })
                await ws.send(sub_msg)
                log.info(f"Subscribed to {len(token_ids)} tokens (reconnect #{reconnect_count})")

                while time.time() < end_time:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=60)
                    except asyncio.TimeoutError:
                        continue

                    recv_ts = time.time()

                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError:
                        continue

                    messages = data if isinstance(data, list) else [data]

                    for msg in messages:
                        if not isinstance(msg, dict):
                            continue

                        event_type = msg.get("event_type", "unknown")
                        record = {"_ts": recv_ts, "_type": event_type}

                        if event_type == "price_change":
                            for ch in msg.get("price_changes", []):
                                rec = {
                                    **record,
                                    "asset_id": ch.get("asset_id"),
                                    "price": ch.get("price"),
                                    "size": ch.get("size"),
                                    "side": ch.get("side"),
                                    "best_bid": ch.get("best_bid"),
                                    "best_ask": ch.get("best_ask"),
                                }
                                _append_jsonl(out_path, rec)
                                event_count += 1

                        elif event_type == "last_trade_price":
                            for trade in msg.get("last_trade_prices", [msg]):
                                rec = {
                                    **record,
                                    "asset_id": trade.get("asset_id"),
                                    "price": trade.get("price"),
                                    "size": trade.get("size"),
                                }
                                _append_jsonl(out_path, rec)
                                event_count += 1

                        elif event_type == "book":

                            bids = msg.get("bids", [])
                            asks = msg.get("asks", [])
                            rec = {
                                **record,
                                "asset_id": msg.get("asset_id"),
                                "market": msg.get("market"),
                                "bids_depth": len(bids),
                                "asks_depth": len(asks),
                                "best_bid": bids[0]["price"] if bids else None,
                                "best_ask": asks[0]["price"] if asks else None,
                            }
                            _append_jsonl(out_path, rec)
                            event_count += 1

                    if event_count % 1000 == 0 and event_count > 0:
                        log.info(f"Collected {event_count:,} events → {out_path.name}")

        except (websockets.exceptions.ConnectionClosed, OSError) as e:
            reconnect_count += 1
            log.warning(f"Connection lost ({e}), reconnecting in {RECONNECT_DELAY}s...")
            await asyncio.sleep(RECONNECT_DELAY)
        except Exception as e:
            log.error(f"Unexpected error: {e}")
            await asyncio.sleep(RECONNECT_DELAY)

    log.info(f"Collection complete: {event_count:,} events in {out_path}")
    return event_count

def main():
    parser = argparse.ArgumentParser(description="Polymarket WebSocket collector")
    parser.add_argument(
        "--tokens-file",
        default="data/processed/top_markets_for_collection.json",
        help="Path to JSON with market list",
    )
    parser.add_argument("--max-tokens", type=int, default=50, help="Max tokens to subscribe")
    parser.add_argument("--duration", type=int, default=3600, help="Collection duration in seconds")
    parser.add_argument("--prefix", default="ws_stream", help="Output file prefix")
    args = parser.parse_args()

    token_ids = load_token_ids(args.tokens_file, args.max_tokens)

    log.info(f"Starting WebSocket collector: {len(token_ids)} tokens, {args.duration}s")
    count = asyncio.run(collect_stream(token_ids, args.duration, args.prefix))
    log.info(f"Finished: {count:,} events collected")

if __name__ == "__main__":
    main()
