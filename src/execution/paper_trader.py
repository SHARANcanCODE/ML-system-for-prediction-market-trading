import json
import signal as signal_mod
import sys
import time
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np

warnings.filterwarnings("ignore", message="X does not have valid feature names")

from src.data.client import PolymarketClient
from src.risk.manager import RiskManager, RiskConfig, Regime
from src.strategies.signal_engine import SignalEngine
from src.strategies.metamodel import StrategyRouter
from src.utils.fees import polymarket_fee as fee
from src.utils.logger import get_logger

log = get_logger(__name__)

PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_LOGS_DIR = PROJECT / "logs" / "paper_trading"
MODELS_DIR = PROJECT / "data" / "models"

LOGS_DIR = DEFAULT_LOGS_DIR
CHECKPOINT_PATH = DEFAULT_LOGS_DIR / "checkpoint.json"

@dataclass
class Position:
    token_id: str
    condition_id: str
    side: Literal["YES", "NO"]
    entry_price: float
    n_shares: float
    entry_fee: float
    entry_time: str
    bet_size: float = 0.0
    category: str = ""
    source: str = ""
    market_question: str = ""
    peak_price: float = 0.0

@dataclass
class Trade:
    token_id: str
    condition_id: str
    side: Literal["YES", "NO"]
    entry_price: float
    exit_price: float
    n_shares: float
    entry_fee: float
    exit_fee: float
    pnl: float
    entry_time: str
    exit_time: str
    exit_reason: str
    source: str = ""
    category: str = ""
    market_question: str = ""

@dataclass
class PaperPortfolio:
    initial_capital: float = 1000.0
    cash: float = 1000.0
    positions: dict = field(default_factory=dict)
    trades: list = field(default_factory=list)
    equity_curve: list = field(default_factory=list)

    @property
    def n_positions(self) -> int:
        return len(self.positions)

    @property
    def total_invested(self) -> float:
        return sum(p.entry_price * p.n_shares for p in self.positions.values())

    @property
    def equity(self) -> float:
        return self.cash + self.total_invested

    def mark_to_market_equity(self, live_prices: dict[str, float]) -> float:
        mtm_value = 0.0
        for tid, pos in self.positions.items():
            yes_mid = live_prices.get(tid)
            if yes_mid is not None:
                token_price = (1.0 - yes_mid) if pos.side == "NO" else yes_mid
            else:
                token_price = pos.entry_price
            mtm_value += token_price * pos.n_shares
        return self.cash + mtm_value

    @property
    def total_pnl(self) -> float:
        return sum(t.pnl for t in self.trades)

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        return sum(1 for t in self.trades if t.pnl > 0) / len(self.trades)

    def snapshot(self) -> dict:
        return {
            "time": datetime.now(timezone.utc).isoformat(),
            "cash": self.cash,
            "n_positions": self.n_positions,
            "invested": self.total_invested,
            "equity": self.equity,
            "total_pnl": self.total_pnl,
            "n_trades": len(self.trades),
            "win_rate": self.win_rate,
        }

class PaperTrader:

    def __init__(
        self,
        model_path: str = "lgb_v3.joblib",
        calibrator_path: str = "calibrators_v3.joblib",
        meta_path: str = "htr_meta_v1.json",
        htr_model_path: str | None = "lgb_true_htr_v1.joblib",
        htr_meta_path: str | None = "htr_v1_meta.json",
        initial_capital: float = 1000.0,
        fee_rate: float = 0.0175,
        edge_threshold: float = 0.01,
        max_position_pct: float = 0.10,
        max_positions: int = 10,
        cash_reserve_pct: float = 0.20,
        poll_interval: int = 300,

        log_dir: str | None = None,
        instance_name: str | None = None,
        price_min: float = 0.10,
        price_max: float = 0.90,
        categories: list[str] | None = None,
        exclude_categories: list[str] | None = None,
        kelly_fraction: float = 0.5,
        time_exit_hours: float = 12.0,
        min_liquidity: float = 50_000,
        max_liquidity: float | None = None,
        small_markets: bool = False,

        inverse_mode: bool = False,
        rule_based_only: bool = False,
        use_meta: bool = False,
        meta_threshold: float = 0.5,

        adverse_move_threshold: float = 0.10,
        trailing_keep_pct: float = 0.50,
        min_entry_price: float = 0.05,
        edge_gone_cooldown_hours: float = 1.0,
        max_hold_by_category: dict | None = None,
    ):

        global LOGS_DIR, CHECKPOINT_PATH
        if log_dir:
            LOGS_DIR = Path(log_dir)
        elif instance_name:
            LOGS_DIR = DEFAULT_LOGS_DIR / instance_name
        else:
            LOGS_DIR = DEFAULT_LOGS_DIR
        CHECKPOINT_PATH = LOGS_DIR / "checkpoint.json"

        self._instance_name = instance_name or "default"

        self._inverse_mode = inverse_mode
        self._rule_based_only = rule_based_only
        self._use_meta = use_meta
        self._meta_threshold = meta_threshold
        self._meta_model = None

        if use_meta:
            meta_path = MODELS_DIR / "meta_minimal_v1.joblib"
            if meta_path.exists():
                import joblib
                self._meta_model = joblib.load(meta_path)
                log.info(f"Meta model loaded: {meta_path}")
            else:
                log.warning(f"Meta model not found at {meta_path}, disabling meta filter")
                self._use_meta = False

        self._price_range = (price_min, price_max)
        self._category_whitelist = set(c.lower() for c in categories) if categories else None
        self._category_blacklist = set(c.lower() for c in exclude_categories) if exclude_categories else None

        self.signal_engine = SignalEngine(
            tb_model_path=model_path,
            tb_calibrator_path=calibrator_path,
            tb_meta_path=meta_path,
            htr_model_path=htr_model_path,
            htr_meta_path=htr_meta_path,
            fee_rate=fee_rate,
            edge_threshold=edge_threshold,
        )

        self.strategy_router = StrategyRouter()

        self.fee_rate = fee_rate
        self.edge_threshold = edge_threshold
        self.max_position_pct = max_position_pct
        self.max_positions = max_positions
        self.cash_reserve_pct = cash_reserve_pct
        self.poll_interval = poll_interval

        self.portfolio = PaperPortfolio(
            initial_capital=initial_capital,
            cash=initial_capital,
        )

        self.risk = RiskManager(RiskConfig(
            max_position_pct=max_position_pct,
            max_positions=max_positions,
            cash_reserve_pct=cash_reserve_pct,
            min_edge=edge_threshold,
            kelly_fraction=kelly_fraction,
            time_exit_hours=time_exit_hours,
        ))
        self.risk.reset_daily(initial_capital)
        self.risk.update_equity(initial_capital)

        self.client = PolymarketClient()

        self._market_meta: dict = {}

        self._category_keywords = {
            "sports": ["nba", "nfl", "nhl", "mlb", "premier league", "champions league",
                       "serie a", "la liga", "bundesliga", "ligue 1", "uefa", "fifa",
                       "world cup", "super bowl", "playoff", "celtics", "knicks", "lakers",
                       "warriors", "oilers", "yankees", "match winner", "spread", "o/u",
                       "soccer", "football", "basketball", "baseball", "hockey", "tennis",
                       "mma", "ufc", "boxing", "f1", "grand prix", "olympics",
                       "cricket", "ipl", "golf", "pga", "rugby", "formula 1",
                       "esports", "lol", "dota", "cs2", "valorant", "league of legends",
                       "march madness", "ncaa", "copa", "euros", "afc", "nfc",
                       "mvp", "ballon d'or", "tottenham", "arsenal", "liverpool",
                       "manchester", "barcelona", "real madrid", "juventus", "bayern",
                       "world series", "stanley cup", "super bowl"],
            "politics": ["trump", "biden", "president", "election", "congress", "senate",
                         "governor", "democrat", "republican", "gop", "potus", "white house",
                         "impeach", "poll", "primary", "cabinet", "supreme court"],
            "crypto": ["bitcoin", "btc", "ethereum", "eth", "solana", "sol", "crypto",
                       "token price", "defi", "nft"],
            "commodity": ["crude oil", "oil price", "wti", "brent", "natural gas",
                          "gold price", "silver price", "commodity", "commodities"],
            "economics": ["fed ", "interest rate", "inflation", "gdp", "unemployment",
                          "tariff", "trade war", "recession", "cpi", "fomc"],
            "geopolitics": ["iran", "russia", "ukraine", "china", "war", "sanctions",
                            "nato", "ceasefire", "invasion", "military"],
        }

        self._adverse_move_threshold = adverse_move_threshold
        self._trailing_keep_pct = trailing_keep_pct
        self._min_entry_price = min_entry_price
        self._edge_gone_cooldown_hours = edge_gone_cooldown_hours
        self._max_hold_by_category = max_hold_by_category or {
            "sports": 6.0,
            "crypto": 6.0,
            "commodity": 6.0,
            "geopolitics": 12.0,
            "economics": 12.0,
            "politics": 24.0,
        }

        self._close_cooldown: dict[str, datetime] = {}
        self._cooldown_minutes = 30

        self._checkpoint_interval = 3
        self._iteration_since_checkpoint = 0

        self._refresh_interval = 15
        self._iterations_since_refresh = 0
        self._min_liquidity = min_liquidity
        self._max_liquidity = max_liquidity
        self._small_markets = small_markets

        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.log_file = LOGS_DIR / f"paper_trades_{ts}.jsonl"
        self.equity_file = LOGS_DIR / f"paper_equity_{ts}.csv"

        log.info(f"PaperTrader [{self._instance_name}] initialized: capital=${initial_capital}, "
                 f"fee={fee_rate:.2%}, edge={edge_threshold}, kelly={kelly_fraction}, "
                 f"price=({price_min}-{price_max}), categories={categories or 'all'}, "
                 f"exclude={exclude_categories or 'none'}, logs={LOGS_DIR}")

    def _log_event(self, event_type: str, data: dict):
        record = {
            "time": datetime.now(timezone.utc).isoformat(),
            "type": event_type,
            **data,
        }
        with open(self.log_file, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def _compute_signal(self, market_data: dict) -> dict | None:
        signal = None

        if not self._rule_based_only:

            ml_signal = self.signal_engine.generate(market_data)
            if ml_signal is not None:
                signal = {
                    "p_model": ml_signal.p_model,
                    "p_market": ml_signal.p_market,
                    "ev": ml_signal.ev,
                    "side": ml_signal.side,
                    "source": ml_signal.source,
                }

        if signal is None:

            route = self.strategy_router.route(market_data)
            if not route.skip:
                signals = self.strategy_router.generate_signals(market_data)
                if signals:
                    best = signals[0]
                    p_market = float(market_data.get("midpoint", 0.5))
                    entry_fee = fee(p_market, self.fee_rate)
                    ev = best.edge - entry_fee
                    if ev > self.edge_threshold:
                        signal = {
                            "p_model": best.p_model,
                            "p_market": p_market,
                            "ev": ev,
                            "side": best.side,
                            "source": best.strategy,
                        }

        if signal is None:
            return None

        if self._inverse_mode and signal["source"] in ("htr", "tb"):
            signal["side"] = "NO" if signal["side"] == "YES" else "YES"
            signal["p_model"] = 1.0 - signal["p_model"]
            signal["source"] = f"inv_{signal['source']}"

        if self._use_meta and self._meta_model is not None and signal["source"] in ("htr", "tb"):
            p = signal["p_model"]
            conf = abs(p - 0.5) * 2
            direction = 1.0 if signal["side"] == "YES" else 0.0
            meta_input = np.array([[p, conf, direction]])
            meta_prob = self._meta_model.predict(meta_input)[0]
            if meta_prob < self._meta_threshold:
                return None
            signal["source"] = f"meta_{signal['source']}"

        return signal

    def _can_open_position(self, token_id: str, market_data: dict = None) -> bool:
        if token_id in self.portfolio.positions:
            return False

        if token_id in self._close_cooldown:
            elapsed = (datetime.now(timezone.utc) - self._close_cooldown[token_id]).total_seconds() / 60
            if elapsed < self._cooldown_minutes:
                return False

        market_data = market_data or {}
        cat = (market_data.get("category") or "").lower().strip()
        if self._category_whitelist and cat not in self._category_whitelist:
            return False
        if self._category_blacklist and cat in self._category_blacklist:
            return False

        ok, reason = self.risk.can_open(
            equity=self.portfolio.equity,
            cash=self.portfolio.cash,
            n_positions=self.portfolio.n_positions,
            token_id=token_id,
            event_id=market_data.get("condition_id", ""),
            market_liquidity=float(market_data.get("liquidity", 0) or 0),
            market_spread=float(market_data.get("spread", 0) or 0),
        )
        if not ok:
            log.debug(f"Position blocked: {reason}")
        return ok

    def open_position(self, token_id: str, condition_id: str,
                      signal: dict, market_data: dict):
        if not self._can_open_position(token_id, market_data):
            return

        bet_size = self.risk.compute_size(
            equity=self.portfolio.equity,
            cash=self.portfolio.cash,
            p_model=signal["p_model"],
            p_market=signal["p_market"],
            fee_rate=self.fee_rate,
        )
        if bet_size <= 0:
            return

        p_market = signal["p_market"]

        if signal["side"] == "NO":
            entry_price = 1.0 - p_market
        else:
            entry_price = p_market

        if entry_price < self._min_entry_price:
            log.debug(f"Skipped penny bet: entry {entry_price:.3f} < {self._min_entry_price}")
            return
        entry_fee_total = fee(entry_price, self.fee_rate)
        n_shares = bet_size / entry_price

        category = market_data.get("category", "")
        pos = Position(
            token_id=token_id,
            condition_id=condition_id,
            side=signal["side"],
            entry_price=entry_price,
            n_shares=n_shares,
            entry_fee=entry_fee_total * n_shares,
            entry_time=datetime.now(timezone.utc).isoformat(),
            bet_size=bet_size,
            category=category,
            source=signal.get("source", ""),
            market_question=market_data.get("question", ""),
            peak_price=entry_price,
        )

        self.portfolio.positions[token_id] = pos
        self.portfolio.cash -= (bet_size + pos.entry_fee)
        self.risk.position_opened(
            event_id=condition_id,
            category=category,
            invested=bet_size,
        )

        self._log_event("OPEN", {
            "token_id": token_id,
            "side": signal["side"],
            "price": entry_price,
            "yes_midpoint": p_market,
            "n_shares": n_shares,
            "bet_size": bet_size,
            "ev": signal["ev"],
            "p_model": signal["p_model"],
            "source": signal.get("source", "unknown"),
            "question": market_data.get("question", "")[:80],
        })

        source = signal.get("source", "")
        log.info(f"OPEN {signal['side']} {n_shares:.1f} shares @ ${entry_price:.3f} "
                 f"(EV={signal['ev']:.4f}, {source}) — {market_data.get('question', '')[:50]}")

    def close_position(self, token_id: str, yes_midpoint: float, reason: str):
        if token_id not in self.portfolio.positions:
            return

        pos = self.portfolio.positions.pop(token_id)

        if pos.side == "NO":
            exit_price = 1.0 - yes_midpoint
        else:
            exit_price = yes_midpoint

        exit_fee_total = fee(exit_price, self.fee_rate) * pos.n_shares

        pnl = (exit_price - pos.entry_price) * pos.n_shares - pos.entry_fee - exit_fee_total

        trade = Trade(
            token_id=token_id,
            condition_id=pos.condition_id,
            side=pos.side,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            n_shares=pos.n_shares,
            entry_fee=pos.entry_fee,
            exit_fee=exit_fee_total,
            pnl=pnl,
            entry_time=pos.entry_time,
            exit_time=datetime.now(timezone.utc).isoformat(),
            exit_reason=reason,
            source=pos.source,
            category=pos.category,
            market_question=pos.market_question,
        )
        self.portfolio.trades.append(trade)
        self.portfolio.cash += (exit_price * pos.n_shares - exit_fee_total)
        self.risk.position_closed(
            event_id=pos.condition_id,
            category=pos.category,
            pnl=pnl,
            invested=pos.bet_size,
            equity=self.portfolio.equity,
        )
        self.risk.update_equity(self.portfolio.equity)

        self._log_event("CLOSE", {
            "token_id": token_id,
            "side": pos.side,
            "entry": pos.entry_price,
            "exit": exit_price,
            "n_shares": pos.n_shares,
            "bet_size": pos.bet_size,
            "pnl": pnl,
            "pnl_pct": pnl / pos.bet_size * 100 if pos.bet_size else 0,
            "hold_time": pos.entry_time,
            "reason": reason,
            "source": pos.source,
            "category": pos.category,
            "question": pos.market_question[:80],
        })

        self._close_cooldown[token_id] = datetime.now(timezone.utc)

        log.info(f"CLOSE {pos.side} @ ${exit_price:.3f} "
                 f"PnL=${pnl:+.2f} ({reason})")

    def check_exits(self, markets: list[dict]):
        live_prices = {m["token_id"]: m["midpoint"] for m in markets
                       if "midpoint" in m and "token_id" in m}

        for token_id, pos in list(self.portfolio.positions.items()):
            current_price = live_prices.get(token_id)
            if current_price is None:
                continue

            source = pos.source
            if source in ("mean_reversion", "contrarian", "event_driven_nlp"):
                regime = Regime.MEAN_REVERTING
            elif source == "momentum":
                regime = Regime.MOMENTUM
            else:

                regime = Regime.MEAN_REVERTING

            token_price = (1.0 - current_price) if pos.side == "NO" else current_price

            if token_price >= 0.995:
                self.close_position(token_id, current_price,
                    f"Resolution win: token price {token_price:.3f}")
                continue
            if token_price <= 0.005:
                self.close_position(token_id, current_price,
                    f"Resolution loss: token price {token_price:.3f}")
                continue

            pos.peak_price = max(pos.peak_price, token_price)

            adverse_move = pos.entry_price - token_price
            if adverse_move > self._adverse_move_threshold:
                self.close_position(token_id, current_price,
                    f"Adverse move exit: {adverse_move:+.3f} from entry {pos.entry_price:.3f}")
                continue

            peak_profit = pos.peak_price - pos.entry_price
            current_profit = token_price - pos.entry_price

            if peak_profit > 0.02:
                if current_profit < peak_profit * self._trailing_keep_pct:
                    self.close_position(token_id, current_price,
                        f"Trailing exit: profit dropped from {peak_profit:+.3f} to {current_profit:+.3f}")
                    continue

                if peak_profit > 0.05 and current_profit < 0:
                    self.close_position(token_id, current_price,
                        f"Trailing emergency: was {peak_profit:+.3f}, now {current_profit:+.3f}")
                    continue

            entry_time = datetime.fromisoformat(pos.entry_time)
            hold_hours = (datetime.now(timezone.utc) - entry_time).total_seconds() / 3600
            cat = pos.category.lower() if pos.category else ""
            cat_max_hold = self._max_hold_by_category.get(cat, self.risk.config.time_exit_hours)
            if hold_hours >= cat_max_hold:
                self.close_position(token_id, current_price,
                    f"Category time exit: held {hold_hours:.0f}h >= {cat_max_hold:.0f}h ({cat or 'default'})")
                continue

            p_model_current = 0.0
            market_data = next((m for m in markets if m.get("token_id") == token_id), None)
            if market_data:
                if source == "tb":
                    ml_signal = self.signal_engine.generate_tb_signal(market_data)
                elif source == "htr":
                    ml_signal = self.signal_engine.generate_htr_signal(market_data)
                else:
                    ml_signal = self.signal_engine.generate(market_data)
                if ml_signal is not None:
                    p_model_current = ml_signal.p_model

            should_exit, reason = self.risk.should_exit(
                regime=regime,
                entry_time=entry_time,
                entry_price=pos.entry_price,
                current_price=token_price,
                p_model_current=p_model_current,
                p_market_current=current_price,
                side=pos.side,
            )
            if should_exit:

                if "Edge gone" in reason and hold_hours < self._edge_gone_cooldown_hours:
                    log.debug(f"Edge gone suppressed (hold {hold_hours:.1f}h < {self._edge_gone_cooldown_hours}h)")
                    continue
                self.close_position(token_id, current_price, reason)

    def save_equity(self, live_prices: dict[str, float] | None = None):
        snap = self.portfolio.snapshot()
        snap["mtm_equity"] = (
            self.portfolio.mark_to_market_equity(live_prices)
            if live_prices else ""
        )
        self.portfolio.equity_curve.append(snap)
        file_empty = not self.equity_file.exists() or self.equity_file.stat().st_size == 0
        with open(self.equity_file, "a") as f:
            if file_empty:
                f.write(",".join(snap.keys()) + "\n")
            f.write(",".join(str(v) for v in snap.values()) + "\n")

    def status(self, live_prices: dict[str, float] | None = None) -> str:
        p = self.portfolio
        risk_status = self.risk.status()
        halted = " [HALTED]" if risk_status["halted"] else ""
        mtm = p.mark_to_market_equity(live_prices) if live_prices else p.equity
        return (
            f"Equity: ${mtm:.2f} | Cash: ${p.cash:.2f} | "
            f"Positions: {p.n_positions} | Trades: {len(p.trades)} | "
            f"PnL: ${p.total_pnl:+.2f} | WR: {p.win_rate:.0%}"
            f" | DayPnL: ${risk_status['daily_pnl']:+.2f}{halted}"
        )

    def run_once(self, markets: list[dict]):

        self.risk.reset_daily(self.portfolio.equity)

        self.check_exits(markets)

        live_prices = {m["token_id"]: m["midpoint"] for m in markets
                       if "midpoint" in m and "token_id" in m}

        ok, reason = self.risk.check_limits(self.portfolio.equity, self.portfolio.n_positions)
        if not ok:
            log.warning(f"Trading halted: {reason}")
            self.save_equity(live_prices)
            return

        for market in markets:
            token_id = market.get("token_id")
            if not token_id:
                continue

            signal = self._compute_signal(market)
            if signal is not None:
                self.open_position(
                    token_id=token_id,
                    condition_id=market.get("condition_id", ""),
                    signal=signal,
                    market_data=market,
                )

        self.save_equity(live_prices)

    def _classify_category(self, question: str, event_title: str = "") -> str:
        text = f"{question} {event_title}".lower()
        for cat, keywords in self._category_keywords.items():
            if any(kw in text for kw in keywords):
                return cat
        return "other"

    def _enrich_from_gamma(self, token_ids: list[str]):
        import httpx
        uncached = set(t for t in token_ids if t not in self._market_meta)
        if not uncached:
            return

        try:
            import json as _json

            found = set()
            for offset in range(0, 500, 100):
                if not (uncached - found):
                    break
                resp = httpx.get(
                    "https://gamma-api.polymarket.com/markets",
                    params={"limit": 100, "offset": offset,
                            "active": True, "closed": False,
                            "order": "volume24hr", "ascending": False},
                    timeout=15,
                )
                if resp.status_code != 200:
                    break
                markets = resp.json()
                if not markets:
                    break
                for m in markets:
                    tokens = _json.loads(m.get("clobTokenIds", "[]"))
                    for tok in tokens:
                        if tok in uncached:
                            question = m.get("question", "")
                            event_title = ""
                            events = m.get("events", [])
                            if events and isinstance(events, list):
                                event_title = events[0].get("title", "")
                            self._market_meta[tok] = {
                                "question": question,
                                "category": self._classify_category(question, event_title),
                                "volume": float(m.get("volumeNum", 0) or 0),
                                "liquidity": float(m.get("liquidityNum", 0) or 0),
                                "neg_risk": m.get("negRisk", False),
                            }
                            found.add(tok)
            log.info(f"Gamma enrichment: {len(self._market_meta)}/{len(token_ids)} tokens cached")
        except Exception as e:
            log.warning(f"Gamma enrichment failed: {e}")

    def _matches_category_filter(self, category: str) -> bool:
        cat = category.lower().strip()
        if self._category_whitelist and cat not in self._category_whitelist:
            return False
        if self._category_blacklist and cat in self._category_blacklist:
            return False
        return True

    def _scan_gamma_markets(self, existing: set[str], max_tokens: int = 30,
                            pages: int = 5) -> list[str]:
        import httpx
        import json as _json

        new_tokens = []
        for offset in range(0, pages * 100, 100):
            if len(new_tokens) + len(existing) >= max_tokens:
                break
            try:
                resp = httpx.get(
                    "https://gamma-api.polymarket.com/markets",
                    params={
                        "limit": 100, "offset": offset,
                        "active": True, "closed": False,
                        "order": "volume24hr",
                        "ascending": self._small_markets,
                    },
                    timeout=15,
                )
                if resp.status_code != 200:
                    break
                markets = resp.json()
                if not markets:
                    break
            except Exception as e:
                log.warning(f"Gamma API error at offset {offset}: {e}")
                break

            for m in markets:
                toks = _json.loads(m.get("clobTokenIds", "[]"))
                if len(toks) < 2:
                    continue
                tok = toks[0]
                if tok in existing:
                    continue

                liq = float(m.get("liquidityNum", 0) or 0)
                if liq < self._min_liquidity:
                    continue
                if self._max_liquidity is not None and liq > self._max_liquidity:
                    continue

                question = m.get("question", "")
                event_title = ""
                events = m.get("events", [])
                if events and isinstance(events, list):
                    event_title = events[0].get("title", "")
                category = self._classify_category(question, event_title)

                if not self._matches_category_filter(category):
                    continue

                try:
                    r = httpx.get(
                        "https://clob.polymarket.com/midpoint",
                        params={"token_id": tok},
                        timeout=10,
                    )
                    mid = float(r.json().get("mid", 0))
                except Exception:
                    continue

                lo, hi = self._price_range
                if lo <= mid <= hi:
                    new_tokens.append(tok)
                    self._market_meta[tok] = {
                        "question": question,
                        "category": category,
                        "volume": float(m.get("volumeNum", 0) or 0),
                        "liquidity": liq,
                        "neg_risk": m.get("negRisk", False),
                    }
                    log.info(f"  Found: p={mid:.3f} liq=${liq:,.0f} [{category}] "
                             f"{question[:60]}")
                    if len(new_tokens) + len(existing) >= max_tokens:
                        break

        return new_tokens

    def discover_markets(self, target: int = 20) -> list[str]:
        liq_range = f"${self._min_liquidity:,.0f}"
        if self._max_liquidity is not None:
            liq_range += f"–${self._max_liquidity:,.0f}"
        mode = " [small-markets]" if self._small_markets else ""
        log.info(f"Auto-discovering markets: price={self._price_range}, "
                 f"categories={self._category_whitelist or 'all'}, "
                 f"exclude={self._category_blacklist or 'none'}, "
                 f"liq={liq_range}{mode}")

        tokens = self._scan_gamma_markets(
            existing=set(), max_tokens=target, pages=5,
        )

        log.info(f"Discovered {len(tokens)} markets")
        return tokens

    def _refresh_markets(self, token_ids: list[str]) -> list[str]:
        cash_pct = self.portfolio.cash / self.portfolio.equity if self.portfolio.equity > 0 else 1.0
        if cash_pct < 0.30:
            return token_ids
        if self.portfolio.n_positions >= self.max_positions:
            return token_ids

        try:

            import httpx
            alive_tokens = []
            dead_count = 0
            for tid in token_ids:
                try:
                    r = httpx.get(
                        "https://clob.polymarket.com/midpoint",
                        params={"token_id": tid}, timeout=5,
                    )
                    mid = float(r.json().get("mid", 0))
                    if mid > 0.005:
                        alive_tokens.append(tid)
                    else:
                        dead_count += 1
                except Exception:
                    alive_tokens.append(tid)

            if dead_count > 0:
                log.info(f"Market refresh: pruned {dead_count} dead tokens "
                         f"({len(token_ids)} → {len(alive_tokens)})")

            existing = set(alive_tokens)
            new_tokens = self._scan_gamma_markets(
                existing, max_tokens=50, pages=3,
            )

            if new_tokens:
                log.info(f"Market refresh: +{len(new_tokens)} new tokens "
                         f"(total {len(existing) + len(new_tokens)}, "
                         f"cash {cash_pct:.0%})")
                return alive_tokens + new_tokens
            else:
                log.debug("Market refresh: no new tokens found")
                return alive_tokens

        except Exception as e:
            log.warning(f"Market refresh failed: {e}")

        return token_ids

    def run(self, token_ids: list[str], duration_minutes: int = 60,
            resume: bool = True):

        resumed = False
        if resume:
            resumed = self._load_checkpoint()

        log.info(f"Starting paper trading: {len(token_ids)} tokens, "
                 f"{duration_minutes}min, poll every {self.poll_interval}s"
                 f"{' (resumed)' if resumed else ''}")

        self._enrich_from_gamma(token_ids)

        self._stop_requested = False
        def _handle_sigterm(signum, frame):
            log.info("Received SIGTERM — saving checkpoint and stopping...")
            self._stop_requested = True
        signal_mod.signal(signal_mod.SIGTERM, _handle_sigterm)

        end_time = time.time() + duration_minutes * 60
        iteration = 0
        last_iteration_time = time.time()

        stop_file = LOGS_DIR / "STOP"

        while time.time() < end_time and not self._stop_requested:

            if stop_file.exists():
                log.info(f"STOP file detected ({stop_file}) — stopping...")
                stop_file.unlink(missing_ok=True)
                break

            iteration += 1
            try:

                gap = time.time() - last_iteration_time
                if gap > self.poll_interval * 2:
                    gap_min = gap / 60
                    log.warning(f"Sleep gap detected: {gap_min:.0f}min since last iteration "
                                f"(expected {self.poll_interval/60:.0f}min)")
                last_iteration_time = time.time()

                self._iterations_since_refresh += 1
                if self._iterations_since_refresh >= self._refresh_interval:
                    token_ids = self._refresh_markets(token_ids)
                    self._iterations_since_refresh = 0

                markets = []
                for tid in token_ids:
                    try:
                        mid = self.client.get_midpoint(tid)
                        book = self.client.get_order_book(tid)
                        bids = book.get("bids", [])
                        asks = book.get("asks", [])
                        best_bid = max((float(b["price"]) for b in bids), default=0)
                        best_ask = min((float(a["price"]) for a in asks), default=0)
                        meta = self._market_meta.get(tid, {})
                        markets.append({
                            "token_id": tid,
                            "condition_id": book.get("market", tid),
                            "midpoint": mid,
                            "best_bid": best_bid,
                            "best_ask": best_ask,
                            "spread": (best_ask - best_bid) if best_bid and best_ask else 0,
                            "question": meta.get("question", ""),
                            "category": meta.get("category", ""),
                            "volume": meta.get("volume", 0),
                            "liquidity": meta.get("liquidity", 0),
                            "neg_risk": meta.get("neg_risk", False),
                        })
                    except Exception as e:
                        log.debug(f"Skip token {tid[:20]}: {e}")

                self.run_once(markets)

                live_prices = {m["token_id"]: m["midpoint"] for m in markets
                               if "midpoint" in m}
                log.info(f"[{iteration}] {self.status(live_prices)}")

                self._iteration_since_checkpoint += 1
                if self._iteration_since_checkpoint >= self._checkpoint_interval:
                    self._save_checkpoint()
                    self._iteration_since_checkpoint = 0

                remaining = end_time - time.time()
                sleep_time = min(self.poll_interval, remaining)
                if sleep_time > 0:
                    time.sleep(sleep_time)

            except KeyboardInterrupt:
                log.info("Paper trading stopped by user")
                break
            except Exception as e:
                log.error(f"Iteration error: {e}")
                self._save_checkpoint()
                time.sleep(10)

        log.info(f"Paper trading complete. {self.status()}")
        self.save_equity()
        self._save_final_report()
        self._save_checkpoint()
        self.client.close()

    def _save_checkpoint(self):
        state = {
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "cash": self.portfolio.cash,
            "initial_capital": self.portfolio.initial_capital,
            "positions": {
                tid: asdict(pos) for tid, pos in self.portfolio.positions.items()
            },
            "trades": [asdict(t) for t in self.portfolio.trades],
            "cooldowns": {
                tid: ts.isoformat() for tid, ts in self._close_cooldown.items()
            },
            "log_file": str(self.log_file),
            "equity_file": str(self.equity_file),
        }

        tmp = CHECKPOINT_PATH.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(state, f, default=str)
        tmp.rename(CHECKPOINT_PATH)
        log.debug("Checkpoint saved")

    def _load_checkpoint(self) -> bool:
        if not CHECKPOINT_PATH.exists():
            return False

        try:
            with open(CHECKPOINT_PATH) as f:
                state = json.load(f)

            self.portfolio.cash = state["cash"]
            self.portfolio.initial_capital = state["initial_capital"]

            self.portfolio.positions = {}
            for tid, pdata in state.get("positions", {}).items():
                self.portfolio.positions[tid] = Position(**pdata)

            self.portfolio.trades = [Trade(**tdata) for tdata in state.get("trades", [])]

            self._close_cooldown = {
                tid: datetime.fromisoformat(ts)
                for tid, ts in state.get("cooldowns", {}).items()
            }

            prev_log = state.get("log_file")
            prev_equity = state.get("equity_file")
            if prev_log and Path(prev_log).exists():
                self.log_file = Path(prev_log)
            if prev_equity and Path(prev_equity).exists():
                self.equity_file = Path(prev_equity)

            self.risk.reset_daily(self.portfolio.equity)
            self.risk.update_equity(self.portfolio.equity)
            for tid, pos in self.portfolio.positions.items():
                self.risk.position_opened(
                    event_id=pos.condition_id,
                    category=pos.category,
                    invested=pos.bet_size,
                )

            saved_at = state.get("saved_at", "?")
            n_pos = len(self.portfolio.positions)
            n_trades = len(self.portfolio.trades)
            log.info(f"Resumed from checkpoint ({saved_at}): "
                     f"cash=${self.portfolio.cash:.2f}, {n_pos} positions, {n_trades} trades")
            return True
        except Exception as e:
            log.warning(f"Failed to load checkpoint: {e} — starting fresh")
            return False

    def _clear_checkpoint(self):
        if CHECKPOINT_PATH.exists():
            CHECKPOINT_PATH.unlink()

    def _save_final_report(self):
        report = {
            "portfolio": self.portfolio.snapshot(),
            "trades": [asdict(t) for t in self.portfolio.trades],
            "open_positions": [
                {**asdict(pos), "token_id": tid}
                for tid, pos in self.portfolio.positions.items()
            ],
            "config": {
                "instance_name": self._instance_name,
                "fee_rate": self.fee_rate,
                "edge_threshold": self.edge_threshold,
                "max_position_pct": self.max_position_pct,
                "max_positions": self.max_positions,
                "cash_reserve_pct": self.cash_reserve_pct,
                "price_range": list(self._price_range),
                "category_whitelist": list(self._category_whitelist) if self._category_whitelist else None,
                "category_blacklist": list(self._category_blacklist) if self._category_blacklist else None,
            },
        }
        report_path = self.log_file.with_suffix(".report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        log.info(f"Report saved: {report_path}")

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Polymarket paper trader")
    parser.add_argument("--config", type=str, help="JSON config file path")
    parser.add_argument("--tokens", nargs="+", help="Token IDs to monitor")
    parser.add_argument("--auto-discover", action="store_true",
                        help="Auto-discover markets via Gamma API (no --tokens needed)")
    parser.add_argument("--max-markets", type=int, default=20,
                        help="Max markets to discover (default: 20)")
    parser.add_argument("--tokens-from", type=str, default=None,
                        help="Copy token list from another instance's token file "
                             "(e.g. 'main' reads logs/paper_trading/tokens.json)")
    parser.add_argument("--duration", type=int, default=60, help="Duration in minutes")
    parser.add_argument("--capital", type=float, default=1000.0, help="Initial capital ($)")
    parser.add_argument("--fee-rate", type=float, default=0.0175, help="Fee rate (maker=0.0175)")
    parser.add_argument("--edge", type=float, default=0.01, help="Min EV edge threshold")
    parser.add_argument("--poll", type=int, default=300, help="Poll interval seconds")
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Resume from checkpoint if available (default)")
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                        help="Start fresh, ignore checkpoint")

    parser.add_argument("--instance", type=str, default=None,
                        help="Instance name (separate log dir per instance)")
    parser.add_argument("--log-dir", type=str, default=None,
                        help="Custom log directory")
    parser.add_argument("--price-min", type=float, default=0.10,
                        help="Min entry price filter (default: 0.10)")
    parser.add_argument("--price-max", type=float, default=0.90,
                        help="Max entry price filter (default: 0.90)")
    parser.add_argument("--categories", nargs="+", default=None,
                        help="Whitelist categories (e.g. sports crypto)")
    parser.add_argument("--exclude-categories", nargs="+", default=None,
                        help="Blacklist categories (e.g. politics)")
    parser.add_argument("--kelly", type=float, default=0.5,
                        help="Kelly fraction (default: 0.5 = half-Kelly)")
    parser.add_argument("--time-exit", type=float, default=12.0,
                        help="Time exit hours (default: 12)")
    parser.add_argument("--min-liquidity", type=float, default=50000,
                        help="Min market liquidity $ (default: 50000)")
    parser.add_argument("--max-liquidity", type=float, default=None,
                        help="Max market liquidity $ — cap for small-markets mode (default: none)")
    parser.add_argument("--small-markets", action="store_true",
                        help="Target less popular markets (ascending volume order)")

    parser.add_argument("--inverse", action="store_true",
                        help="Inverse mode: flip ML signal direction (sanity check)")
    parser.add_argument("--rule-based-only", action="store_true",
                        help="Disable ML, use only rule-based strategies")
    parser.add_argument("--use-meta", action="store_true",
                        help="Filter ML signals through meta-labeling model")
    parser.add_argument("--meta-threshold", type=float, default=0.5,
                        help="Meta model threshold (default: 0.5)")

    parser.add_argument("--adverse-threshold", type=float, default=0.10,
                        help="Adverse move exit threshold (default: 0.10, was 0.15)")
    parser.add_argument("--min-entry-price", type=float, default=0.05,
                        help="Min entry price — skip penny bets (default: 0.05)")
    parser.add_argument("--edge-gone-cooldown", type=float, default=1.0,
                        help="Hours before edge-gone exit activates (default: 1.0)")
    args = parser.parse_args()

    config = {}
    if args.config:
        with open(args.config) as f:
            config = json.load(f)

    token_ids = args.tokens or config.get("token_ids", [])

    trader = PaperTrader(
        model_path=config.get("model_path", None),
        calibrator_path=config.get("calibrator_path", "calibrators_v3.joblib"),
        meta_path=config.get("meta_path", "htr_meta_v1.json"),
        htr_model_path=config.get("htr_model_path", "lgb_true_htr_v1.joblib"),
        htr_meta_path=config.get("htr_meta_path", "htr_v1_meta.json"),
        initial_capital=args.capital,
        fee_rate=args.fee_rate,
        edge_threshold=args.edge,
        max_position_pct=config.get("max_position_pct", 0.10),
        max_positions=config.get("max_positions", 15),
        cash_reserve_pct=config.get("cash_reserve_pct", 0.20),
        poll_interval=args.poll,

        log_dir=args.log_dir or config.get("log_dir"),
        instance_name=args.instance or config.get("instance_name"),
        price_min=config.get("price_min", args.price_min),
        price_max=config.get("price_max", args.price_max),
        categories=args.categories or config.get("categories"),
        exclude_categories=args.exclude_categories or config.get("exclude_categories"),
        kelly_fraction=config.get("kelly_fraction", args.kelly),
        time_exit_hours=config.get("time_exit_hours", args.time_exit),
        min_liquidity=config.get("min_liquidity", args.min_liquidity),
        max_liquidity=config.get("max_liquidity", args.max_liquidity),
        small_markets=config.get("small_markets", args.small_markets),

        inverse_mode=args.inverse,
        rule_based_only=args.rule_based_only,
        use_meta=args.use_meta,
        meta_threshold=args.meta_threshold,

        adverse_move_threshold=config.get("adverse_move_threshold", args.adverse_threshold),
        min_entry_price=config.get("min_entry_price", args.min_entry_price),
        edge_gone_cooldown_hours=config.get("edge_gone_cooldown", args.edge_gone_cooldown),
    )

    if args.tokens_from and not token_ids:
        source = args.tokens_from

        candidates = []
        if source in ("main", "default"):
            candidates = [
                DEFAULT_LOGS_DIR / "tokens.json",
                DEFAULT_LOGS_DIR / "main" / "tokens.json",
            ]
        else:
            candidates = [
                DEFAULT_LOGS_DIR / source / "tokens.json",
            ]
        loaded = False
        for tf in candidates:
            if tf.exists():
                token_ids = json.load(open(tf))
                log.info(f"Loaded {len(token_ids)} tokens from {tf}")
                loaded = True
                break
        if not loaded:
            log.warning(f"Token file not found for '{source}', falling back to auto-discover")

    if args.auto_discover or not token_ids:
        discovered = trader.discover_markets(target=args.max_markets)
        if not discovered and not token_ids:
            log.error("No markets found and no --tokens provided. Exiting.")
            sys.exit(1)
        token_ids = list(set(token_ids + discovered))

    tokens_file = LOGS_DIR / "tokens.json"
    with open(tokens_file, "w") as f:
        json.dump(token_ids, f)
    log.info(f"Token list saved: {tokens_file} ({len(token_ids)} tokens)")

    trader.run(token_ids=token_ids, duration_minutes=args.duration,
               resume=args.resume)

if __name__ == "__main__":
    main()
