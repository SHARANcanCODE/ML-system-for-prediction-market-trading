import argparse
import json

import pandas as pd

from src.data.client import PolymarketClient
from src.utils.logger import get_logger

log = get_logger(__name__)

def find_tradeable(
    min_volume: float = 10_000,
    max_spread: float = 0.05,
    min_liquidity: float = 1_000,
    price_range: tuple = (0.05, 0.95),
    top_n: int = 50,
) -> list[dict]:
    client = PolymarketClient()

    log.info("Fetching active events...")
    events = []
    for offset in range(0, 1000, 100):
        batch = client.get_events(limit=100, offset=offset, active=True)
        if not batch:
            break
        events.extend(batch)
    log.info(f"Fetched {len(events)} events")

    markets = []
    for event in events:
        for market in event.get("markets", []):
            try:
                volume = float(market.get("volumeNum", 0) or 0)
                liquidity = float(market.get("liquidityClob", 0) or 0)
                spread = float(market.get("spread", 1) or 1)

                prices_raw = market.get("outcomePrices", "[]")
                if isinstance(prices_raw, str):
                    prices = json.loads(prices_raw)
                else:
                    prices = prices_raw
                yes_price = float(prices[0]) if prices else 0.5

                tokens_raw = market.get("clobTokenIds", "[]")
                if isinstance(tokens_raw, str):
                    tokens = json.loads(tokens_raw)
                else:
                    tokens = tokens_raw

                if not tokens:
                    continue

                if volume < min_volume:
                    continue
                if spread > max_spread:
                    continue
                if liquidity < min_liquidity:
                    continue
                if not (price_range[0] <= yes_price <= price_range[1]):
                    continue

                markets.append({
                    "condition_id": market.get("conditionId", ""),
                    "token_id_yes": tokens[0] if len(tokens) > 0 else "",
                    "token_id_no": tokens[1] if len(tokens) > 1 else "",
                    "question": market.get("question", "")[:80],
                    "yes_price": yes_price,
                    "volume": volume,
                    "liquidity": liquidity,
                    "spread": spread,
                    "neg_risk": market.get("negRisk", False),
                })
            except (ValueError, IndexError, json.JSONDecodeError):
                continue

    markets.sort(key=lambda m: m["volume"], reverse=True)
    markets = markets[:top_n]

    log.info(f"Found {len(markets)} tradeable markets")
    return markets

def main():
    parser = argparse.ArgumentParser(description="Find tradeable Polymarket markets")
    parser.add_argument("--top", type=int, default=50, help="Number of top markets")
    parser.add_argument("--min-volume", type=float, default=10_000)
    parser.add_argument("--max-spread", type=float, default=0.05)
    parser.add_argument("--min-liquidity", type=float, default=1_000)
    parser.add_argument("--output", type=str, help="Save to JSON file")
    parser.add_argument("--tokens-only", action="store_true", help="Print only token IDs")
    args = parser.parse_args()

    markets = find_tradeable(
        min_volume=args.min_volume,
        max_spread=args.max_spread,
        min_liquidity=args.min_liquidity,
        top_n=args.top,
    )

    if args.tokens_only:
        for m in markets:
            print(m["token_id_yes"])
    else:
        for i, m in enumerate(markets):
            print(f"{i+1:3d}. [{m['yes_price']:.2f}] vol=${m['volume']:>12,.0f} "
                  f"liq=${m['liquidity']:>8,.0f} spr={m['spread']:.3f} "
                  f"{m['question']}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(markets, f, indent=2)
        log.info(f"Saved to {args.output}")

    token_ids = [m["token_id_yes"] for m in markets if m["token_id_yes"]]
    print(f"\n--- {len(token_ids)} token IDs for paper_trader ---")
    print(" ".join(token_ids[:10]))

if __name__ == "__main__":
    main()
