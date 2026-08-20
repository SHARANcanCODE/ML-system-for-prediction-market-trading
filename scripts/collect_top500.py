import json
import sys
import time

sys.path.insert(0, ".")

from src.data.client import PolymarketClient
from src.data.collectors import collect_prices_for_markets, collect_trades_for_markets
from src.utils.logger import get_logger

log = get_logger("collect_top500")

MARKETS_FILE = "data/processed/top_markets_for_collection.json"

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    limit = 50
    for i, arg in enumerate(sys.argv):
        if arg == "--limit" and i + 1 < len(sys.argv):
            limit = int(sys.argv[i + 1])

    with open(MARKETS_FILE) as f:
        markets = json.load(f)[:limit]
    log.info(f"Using top {len(markets)} markets")

    client = PolymarketClient()

    now = int(time.time())
    start_ts = now - 90 * 24 * 3600

    try:
        if mode in ("trades", "all"):
            log.info("Collecting trades...")
            trades = collect_trades_for_markets(
                client, markets, max_trades_per_market=5000, delay=0.1
            )
            log.info(f"Trades done: {sum(len(t) for t in trades.values())} total trades")

        if mode in ("prices", "all"):
            log.info(f"Collecting price history: fidelity=60, 90 days")
            prices = collect_prices_for_markets(
                client, markets, start_ts, now, fidelity=60, delay=0.1
            )
            log.info(f"Prices done: {len(prices)} tokens")
    finally:
        client.close()

    log.info("Collection complete")

if __name__ == "__main__":
    main()
