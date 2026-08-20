import json
import os
from typing import Any

import httpx
from dotenv import load_dotenv
from py_clob_client.client import ClobClient

from src.utils.logger import get_logger

load_dotenv()
log = get_logger(__name__)

CLOB_URL = "https://clob.polymarket.com"
GAMMA_URL = "https://gamma-api.polymarket.com"
DATA_URL = "https://data-api.polymarket.com"

class PolymarketClient:

    def __init__(self, private_key: str | None = None):
        self.clob = ClobClient(CLOB_URL)
        self._http = httpx.Client(timeout=30)
        self._authenticated = False

        if private_key:
            self.clob = ClobClient(
                CLOB_URL,
                key=private_key,
                chain_id=137,
                signature_type=1,
                funder=os.getenv("POLYMARKET_FUNDER_ADDRESS", ""),
            )
            self.clob.set_api_creds(self.clob.create_or_derive_api_creds())
            self._authenticated = True
            log.info("Authenticated client initialized")
        else:
            log.info("Read-only client initialized")

    def health_check(self) -> bool:
        checks = {}
        try:
            checks["clob"] = self.clob.get_ok() == "OK"
        except Exception as e:
            checks["clob"] = False
            log.error(f"CLOB health check failed: {e}")

        try:
            r = self._http.get(f"{GAMMA_URL}/events", params={"limit": "1"})
            checks["gamma"] = r.status_code == 200
        except Exception as e:
            checks["gamma"] = False
            log.error(f"Gamma health check failed: {e}")

        try:
            r = self._http.get(f"{DATA_URL}/")
            checks["data"] = r.status_code == 200
        except Exception as e:
            checks["data"] = False
            log.error(f"Data API health check failed: {e}")

        log.info(f"Health check: {checks}")
        return all(checks.values())

    def get_events(
        self,
        active: bool = True,
        closed: bool = False,
        limit: int = 100,
        offset: int = 0,
        order: str = "volume24hr",
        ascending: bool = False,
        **kwargs: Any,
    ) -> list[dict]:
        params = {
            "active": str(active).lower(),
            "closed": str(closed).lower(),
            "limit": str(limit),
            "offset": str(offset),
            "order": order,
            "ascending": str(ascending).lower(),
            **{k: str(v) for k, v in kwargs.items()},
        }
        r = self._http.get(f"{GAMMA_URL}/events", params=params)
        r.raise_for_status()
        return r.json()

    def get_event_by_slug(self, slug: str) -> dict | None:
        r = self._http.get(f"{GAMMA_URL}/events/slug/{slug}")
        if r.status_code == 200:
            return r.json()
        return None

    def search(self, query: str) -> dict:
        r = self._http.get(
            f"{GAMMA_URL}/public-search", params={"query": query}
        )
        r.raise_for_status()
        return r.json()

    def get_order_book(self, token_id: str) -> dict:
        book = self.clob.get_order_book(token_id)
        return book.__dict__ if hasattr(book, "__dict__") else book

    def get_midpoint(self, token_id: str) -> float:
        result = self.clob.get_midpoint(token_id)
        if isinstance(result, dict):
            return float(result.get("mid", 0))
        return float(result)

    def get_price(self, token_id: str, side: str = "BUY") -> float:
        result = self.clob.get_price(token_id, side)
        if isinstance(result, dict):
            return float(result.get("price", 0))
        return float(result)

    def get_price_history(
        self,
        token_id: str,
        start_ts: int,
        end_ts: int,
        fidelity: int = 60,
    ) -> list[dict]:
        r = self._http.get(
            f"{CLOB_URL}/prices-history",
            params={
                "market": token_id,
                "startTs": str(start_ts),
                "endTs": str(end_ts),
                "fidelity": str(fidelity),
            },
        )
        r.raise_for_status()
        data = r.json()
        return data.get("history", [])

    def get_trades(
        self,
        market: str | None = None,
        limit: int = 100,
        **kwargs: Any,
    ) -> list[dict]:
        params: dict[str, str] = {"limit": str(limit)}
        if market:
            params["market"] = market
        params.update({k: str(v) for k, v in kwargs.items()})

        r = self._http.get(f"{DATA_URL}/trades", params=params)
        r.raise_for_status()
        return r.json()

    def get_comments(
        self,
        limit: int = 100,
        offset: int = 0,
        order: str = "createdAt",
        ascending: bool = False,
        **kwargs: Any,
    ) -> list[dict]:
        params = {
            "limit": str(limit),
            "offset": str(offset),
            "order": order,
            "ascending": str(ascending).lower(),
            **{k: str(v) for k, v in kwargs.items()},
        }
        r = self._http.get(f"{GAMMA_URL}/comments", params=params)
        r.raise_for_status()
        return r.json()

    def get_event_comment_count(self, event_id: str) -> int:
        r = self._http.get(f"{GAMMA_URL}/events/{event_id}/comments/count")
        r.raise_for_status()
        data = r.json()
        return data if isinstance(data, int) else data.get("count", 0)

    def get_market_holders(self, market: str, limit: int = 20) -> list[dict]:
        r = self._http.get(
            f"{DATA_URL}/holders",
            params={"market": market, "limit": str(limit)},
        )
        r.raise_for_status()
        return r.json()

    def close(self):
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
