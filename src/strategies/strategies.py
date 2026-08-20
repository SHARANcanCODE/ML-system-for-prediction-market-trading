from dataclasses import dataclass
from typing import Literal

import numpy as np

from src.risk.manager import Regime
from src.utils.logger import get_logger

log = get_logger(__name__)

@dataclass
class StrategySignal:
    token_id: str
    condition_id: str
    side: Literal["YES", "NO"]
    p_model: float
    p_market: float
    edge: float
    strategy: str
    regime: Regime
    confidence: float
    meta: dict = None

    def __post_init__(self):
        if self.meta is None:
            self.meta = {}

from src.utils.fees import polymarket_fee as fee

class MeanReversionStrategy:

    def __init__(
        self,
        z_entry: float = 2.0,
        z_exit: float = 0.5,
        min_bars: int = 20,
        fee_rate: float = 0.0175,
        min_edge: float = 0.02,
    ):
        self.z_entry = z_entry
        self.z_exit = z_exit
        self.min_bars = min_bars
        self.fee_rate = fee_rate
        self.min_edge = min_edge

    def generate(self, market_data: dict) -> StrategySignal | None:
        z = market_data.get("z_score")
        p_market = market_data.get("midpoint")
        p_mean = market_data.get("price_mean")
        n_bars = market_data.get("n_bars", 0)

        if z is None or p_market is None or p_mean is None:
            return None
        if n_bars < self.min_bars:
            return None

        if z > self.z_entry:

            side = "NO"
            p_model = 1.0 - p_mean
            edge = p_model - (1.0 - p_market)
        elif z < -self.z_entry:

            side = "YES"
            p_model = p_mean
            edge = p_model - p_market
        else:
            return None

        entry_fee = fee(p_market, self.fee_rate)
        net_edge = abs(edge) - entry_fee
        if net_edge < self.min_edge:
            return None

        return StrategySignal(
            token_id=market_data.get("token_id", ""),
            condition_id=market_data.get("condition_id", ""),
            side=side,
            p_model=p_model,
            p_market=p_market,
            edge=edge,
            strategy="mean_reversion",
            regime=Regime.MEAN_REVERTING,
            confidence=min(abs(z) / 4.0, 1.0),
            meta={"z_score": z, "price_mean": p_mean},
        )

class MomentumStrategy:

    def __init__(
        self,
        lookback_hours: int = 24,
        min_return: float = 0.05,
        fee_rate: float = 0.0175,
        min_edge: float = 0.02,
    ):
        self.lookback_hours = lookback_hours
        self.min_return = min_return
        self.fee_rate = fee_rate
        self.min_edge = min_edge

    def generate(self, market_data: dict) -> StrategySignal | None:
        p_market = market_data.get("midpoint")
        ret = market_data.get("ret_24h", 0.0)
        vol_ratio = market_data.get("volume_ratio", 1.0)

        if p_market is None or ret is None:
            return None

        if abs(ret) < self.min_return:
            return None
        if vol_ratio < 1.2:
            return None

        if ret > 0:
            side = "YES"

            p_model = min(p_market + ret * 0.5, 0.95)
            edge = p_model - p_market
        else:
            side = "NO"
            p_model = max(p_market + ret * 0.5, 0.05)
            edge = (1 - p_model) - (1 - p_market)

        entry_fee = fee(p_market, self.fee_rate)
        net_edge = abs(edge) - entry_fee
        if net_edge < self.min_edge:
            return None

        return StrategySignal(
            token_id=market_data.get("token_id", ""),
            condition_id=market_data.get("condition_id", ""),
            side=side,
            p_model=p_model,
            p_market=p_market,
            edge=edge,
            strategy="momentum",
            regime=Regime.MOMENTUM,
            confidence=min(abs(ret) / 0.15, 1.0),
            meta={"ret_24h": ret, "volume_ratio": vol_ratio},
        )

class ResolutionConvergenceStrategy:

    def __init__(
        self,
        high_prob_threshold: float = 0.90,
        discount_threshold: float = 0.03,
        fee_rate: float = 0.0175,
    ):
        self.high_prob_threshold = high_prob_threshold
        self.discount_threshold = discount_threshold
        self.fee_rate = fee_rate

    def generate(self, market_data: dict) -> StrategySignal | None:
        p_market = market_data.get("midpoint")
        p_model = market_data.get("p_model")
        hours_left = market_data.get("time_to_resolution_hours")

        if p_market is None or p_model is None:
            return None

        if p_model >= self.high_prob_threshold and p_market < p_model - self.discount_threshold:
            side = "YES"
            edge = p_model - p_market
            entry_fee = fee(p_market, self.fee_rate)
            if edge - entry_fee < 0.01:
                return None

            return StrategySignal(
                token_id=market_data.get("token_id", ""),
                condition_id=market_data.get("condition_id", ""),
                side=side,
                p_model=p_model,
                p_market=p_market,
                edge=edge,
                strategy="convergence",
                regime=Regime.MEAN_REVERTING,
                confidence=min(p_model, 1.0),
                meta={"hours_left": hours_left, "discount": edge},
            )

        if p_model <= (1 - self.high_prob_threshold) and p_market > p_model + self.discount_threshold:
            side = "NO"
            edge = p_market - p_model
            entry_fee = fee(1 - p_market, self.fee_rate)
            if edge - entry_fee < 0.01:
                return None

            return StrategySignal(
                token_id=market_data.get("token_id", ""),
                condition_id=market_data.get("condition_id", ""),
                side=side,
                p_model=p_model,
                p_market=p_market,
                edge=edge,
                strategy="convergence",
                regime=Regime.MEAN_REVERTING,
                confidence=min(1 - p_model, 1.0),
                meta={"hours_left": hours_left, "discount": edge},
            )

        return None

class ContrarianStrategy:

    def __init__(
        self,
        yes_bias: float = 0.02,
        longshot_threshold: float = 0.20,
        fee_rate: float = 0.0175,
        min_edge: float = 0.02,
    ):
        self.yes_bias = yes_bias
        self.longshot_threshold = longshot_threshold
        self.fee_rate = fee_rate
        self.min_edge = min_edge

    def generate(self, market_data: dict) -> StrategySignal | None:
        p_market = market_data.get("midpoint")
        cal_error = market_data.get("calibration_error")

        if p_market is None:
            return None

        if cal_error is not None and cal_error > self.yes_bias:
            side = "NO"
            p_model = p_market - cal_error
            edge = cal_error
            entry_fee = fee(1 - p_market, self.fee_rate)
            if edge - entry_fee < self.min_edge:
                return None

            return StrategySignal(
                token_id=market_data.get("token_id", ""),
                condition_id=market_data.get("condition_id", ""),
                side=side,
                p_model=p_model,
                p_market=p_market,
                edge=edge,
                strategy="contrarian",
                regime=Regime.MEAN_REVERTING,
                confidence=min(cal_error / 0.10, 1.0),
                meta={"calibration_error": cal_error, "type": "yes_bias"},
            )

        return None

class NegRiskArbStrategy:

    def __init__(
        self,
        min_deviation: float = 0.02,
        fee_rate: float = 0.0175,
    ):
        self.min_deviation = min_deviation
        self.fee_rate = fee_rate

    def generate(self, market_data: dict) -> list[StrategySignal]:
        outcomes = market_data.get("outcomes", [])
        if len(outcomes) < 2:
            return []

        total = sum(o.get("midpoint", 0) for o in outcomes)
        deviation = total - 1.0

        if abs(deviation) < self.min_deviation:
            return []

        signals = []
        if deviation > 0:

            for o in outcomes:
                p = o.get("midpoint", 0)
                if p > 0.1:
                    edge = deviation / len(outcomes)
                    entry_fee = fee(1 - p, self.fee_rate)
                    if edge - entry_fee > 0.005:
                        signals.append(StrategySignal(
                            token_id=o.get("token_id", ""),
                            condition_id=o.get("condition_id", ""),
                            side="NO",
                            p_model=p - edge,
                            p_market=p,
                            edge=edge,
                            strategy="negrisk_arb",
                            regime=Regime.MEAN_REVERTING,
                            confidence=min(abs(deviation) / 0.05, 1.0),
                            meta={"sum": total, "deviation": deviation},
                        ))
        else:

            for o in outcomes:
                p = o.get("midpoint", 0)
                if p < 0.9:
                    edge = abs(deviation) / len(outcomes)
                    entry_fee = fee(p, self.fee_rate)
                    if edge - entry_fee > 0.005:
                        signals.append(StrategySignal(
                            token_id=o.get("token_id", ""),
                            condition_id=o.get("condition_id", ""),
                            side="YES",
                            p_model=p + edge,
                            p_market=p,
                            edge=edge,
                            strategy="negrisk_arb",
                            regime=Regime.MEAN_REVERTING,
                            confidence=min(abs(deviation) / 0.05, 1.0),
                            meta={"sum": total, "deviation": deviation},
                        ))

        return signals

class MarketMakingStrategy:

    def __init__(
        self,
        min_spread: float = 0.02,
        quote_offset: float = 0.01,
        max_inventory: float = 500.0,
        fee_rate: float = 0.0175,
        min_edge: float = 0.005,
        skew_factor: float = 0.5,
    ):
        self.min_spread = min_spread
        self.quote_offset = quote_offset
        self.max_inventory = max_inventory
        self.fee_rate = fee_rate
        self.min_edge = min_edge
        self.skew_factor = skew_factor

    def generate(self, market_data: dict) -> list[StrategySignal]:
        bid = market_data.get("best_bid")
        ask = market_data.get("best_ask")
        mid = market_data.get("midpoint")

        if bid is None or ask is None or mid is None:
            return []

        spread = ask - bid
        if spread < self.min_spread:
            return []

        entry_fee = fee(mid, self.fee_rate)
        exit_fee = fee(mid, self.fee_rate)
        round_trip_cost = entry_fee + exit_fee

        half_spread = spread / 2
        expected_profit = half_spread - round_trip_cost
        if expected_profit < self.min_edge:
            return []

        inv_yes = market_data.get("inventory_yes", 0.0)
        inv_no = market_data.get("inventory_no", 0.0)
        net_inventory = inv_yes - inv_no
        skew = self.skew_factor * net_inventory / max(self.max_inventory, 1.0)
        skew = np.clip(skew, -0.03, 0.03)

        vol = market_data.get("volatility", 0.0)
        vol_adj = min(vol * 0.5, 0.02)

        our_bid = mid - self.quote_offset - vol_adj - skew
        our_ask = mid + self.quote_offset + vol_adj - skew

        our_bid = np.clip(our_bid, 0.01, 0.99)
        our_ask = np.clip(our_ask, 0.01, 0.99)

        if our_bid >= our_ask:
            return []

        signals = []
        confidence = min(expected_profit / 0.02, 1.0)

        if inv_yes < self.max_inventory:
            signals.append(StrategySignal(
                token_id=market_data.get("token_id", ""),
                condition_id=market_data.get("condition_id", ""),
                side="YES",
                p_model=mid,
                p_market=our_bid,
                edge=expected_profit,
                strategy="market_making",
                regime=Regime.MEAN_REVERTING,
                confidence=confidence,
                meta={
                    "quote_type": "bid",
                    "spread": spread,
                    "our_bid": our_bid,
                    "our_ask": our_ask,
                    "round_trip_cost": round_trip_cost,
                    "skew": skew,
                },
            ))

        if inv_no < self.max_inventory:
            signals.append(StrategySignal(
                token_id=market_data.get("token_id", ""),
                condition_id=market_data.get("condition_id", ""),
                side="NO",
                p_model=1 - mid,
                p_market=1 - our_ask,
                edge=expected_profit,
                strategy="market_making",
                regime=Regime.MEAN_REVERTING,
                confidence=confidence,
                meta={
                    "quote_type": "ask",
                    "spread": spread,
                    "our_bid": our_bid,
                    "our_ask": our_ask,
                    "round_trip_cost": round_trip_cost,
                    "skew": skew,
                },
            ))

        return signals

class EventDrivenNLPStrategy:

    def __init__(
        self,
        sentiment_threshold: float = -0.15,
        min_comments: int = 5,
        velocity_boost: float = 0.3,
        fee_rate: float = 0.0175,
        min_edge: float = 0.02,
    ):
        self.sentiment_threshold = sentiment_threshold
        self.min_comments = min_comments
        self.velocity_boost = velocity_boost
        self.fee_rate = fee_rate
        self.min_edge = min_edge

    def generate(self, market_data: dict) -> StrategySignal | None:
        p_market = market_data.get("midpoint")
        sentiment = market_data.get("nlp_comment_sentiment")
        n_comments = market_data.get("nlp_comment_count", 0)
        velocity = market_data.get("nlp_comment_velocity", 0.0)
        bullish_ratio = market_data.get("nlp_bullish_keyword_ratio", 0.5)

        if p_market is None or sentiment is None:
            return None
        if n_comments < self.min_comments:
            return None

        if sentiment < self.sentiment_threshold:
            side = "YES"

            raw_edge = abs(sentiment) * 0.15

            if bullish_ratio < 0.4:
                raw_edge *= 1.2

        elif sentiment > abs(self.sentiment_threshold) and p_market > 0.60:
            side = "NO"
            raw_edge = sentiment * 0.10
        else:
            return None

        entry_price = p_market if side == "YES" else (1 - p_market)
        entry_fee = fee(entry_price, self.fee_rate)
        net_edge = raw_edge - entry_fee
        if net_edge < self.min_edge:
            return None

        base_confidence = min(abs(sentiment) / 0.5, 0.8)
        vel_boost = min(velocity / 10.0, 1.0) * self.velocity_boost
        confidence = min(base_confidence + vel_boost, 1.0)

        if side == "YES":
            p_model = p_market + raw_edge
        else:
            p_model = p_market - raw_edge

        p_model = np.clip(p_model, 0.01, 0.99)

        return StrategySignal(
            token_id=market_data.get("token_id", ""),
            condition_id=market_data.get("condition_id", ""),
            side=side,
            p_model=p_model,
            p_market=p_market,
            edge=raw_edge,
            strategy="event_driven_nlp",
            regime=Regime.MEAN_REVERTING,
            confidence=confidence,
            meta={
                "sentiment": sentiment,
                "n_comments": n_comments,
                "velocity": velocity,
                "bullish_ratio": bullish_ratio,
                "signal_type": "contrarian_sentiment",
            },
        )
