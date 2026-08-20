from dataclasses import dataclass, field
from typing import Literal

from src.strategies.strategies import (
    ContrarianStrategy,
    EventDrivenNLPStrategy,
    MarketMakingStrategy,
    MeanReversionStrategy,
    MomentumStrategy,
    NegRiskArbStrategy,
    ResolutionConvergenceStrategy,
    StrategySignal,
)
from src.utils.logger import get_logger

log = get_logger(__name__)

@dataclass
class RouteDecision:
    strategy_name: str
    order_type: Literal["maker", "taker"]
    priority: int
    reason: str
    skip: bool = False
    skip_reason: str = ""

@dataclass
class StrategyRouter:

    price_sweet_spot: float = 0.20
    price_marginal_upper: float = 0.50
    price_danger_zone: float = 0.70

    min_liquidity: float = 500.0

    _strategies: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self._strategies = {
            "mean_reversion": MeanReversionStrategy(),
            "momentum": MomentumStrategy(),
            "convergence": ResolutionConvergenceStrategy(),
            "contrarian": ContrarianStrategy(),
            "negrisk_arb": NegRiskArbStrategy(),
            "market_making": MarketMakingStrategy(),
            "event_driven_nlp": EventDrivenNLPStrategy(),
        }

    def route(self, market_data: dict) -> RouteDecision:
        p = market_data.get("midpoint")
        liq = market_data.get("liquidity", 0)

        if p is None:
            return RouteDecision("none", "maker", 99, "", skip=True,
                                 skip_reason="No price data")

        if liq > 0 and liq < self.min_liquidity:
            return RouteDecision("none", "maker", 99, "", skip=True,
                                 skip_reason=f"Low liquidity: ${liq:.0f}")

        if market_data.get("cat_is_culture", 0) and p > self.price_danger_zone:
            return RouteDecision("none", "maker", 99, "", skip=True,
                                 skip_reason="Culture + high price = negative edge")

        if market_data.get("is_negrisk") and market_data.get("outcomes"):
            signals = self._strategies["negrisk_arb"].generate(market_data)
            if signals:
                return RouteDecision("negrisk_arb", "maker", 1,
                                     f"NegRisk arb: {len(signals)} signals")

        if market_data.get("p_model") is not None:
            sig = self._strategies["convergence"].generate(market_data)
            if sig:
                return RouteDecision("convergence", "maker", 2,
                                     f"Convergence: edge={sig.edge:.3f}")

        if p <= self.price_sweet_spot and market_data.get("z_score") is not None:
            sig = self._strategies["mean_reversion"].generate(market_data)
            if sig:
                return RouteDecision("mean_reversion", "maker", 3,
                                     f"MR at low price: z={market_data['z_score']:.2f}")

        if market_data.get("calibration_error") is not None:
            sig = self._strategies["contrarian"].generate(market_data)
            if sig:
                return RouteDecision("contrarian", "maker", 4,
                                     f"Contrarian: cal_err={market_data['calibration_error']:.3f}")

        if market_data.get("nlp_comment_sentiment") is not None:
            sig = self._strategies["event_driven_nlp"].generate(market_data)
            if sig:
                return RouteDecision("event_driven_nlp", "maker", 5,
                                     f"NLP contrarian: sent={market_data['nlp_comment_sentiment']:.2f}")

        if market_data.get("z_score") is not None:
            sig = self._strategies["mean_reversion"].generate(market_data)
            if sig:
                return RouteDecision("mean_reversion", "maker", 6,
                                     f"MR: z={market_data['z_score']:.2f}")

        if market_data.get("ret_24h") is not None:
            sig = self._strategies["momentum"].generate(market_data)
            if sig:
                return RouteDecision("momentum", "maker", 7,
                                     f"Momentum: ret={market_data['ret_24h']:.3f}")

        if market_data.get("best_bid") is not None and market_data.get("best_ask") is not None:
            signals = self._strategies["market_making"].generate(market_data)
            if signals:
                return RouteDecision("market_making", "maker", 8,
                                     f"MM: spread={market_data.get('best_ask', 0) - market_data.get('best_bid', 0):.3f}")

        zone = "sweet" if p <= self.price_sweet_spot else \
               "marginal" if p <= self.price_marginal_upper else \
               "danger" if p <= self.price_danger_zone else "extreme"
        return RouteDecision("none", "maker", 99, "", skip=True,
                             skip_reason=f"No signal (price zone: {zone})")

    def generate_signals(self, market_data: dict) -> list[StrategySignal]:
        signals = []

        for name, strategy in self._strategies.items():
            if name == "negrisk_arb":
                if market_data.get("is_negrisk") and market_data.get("outcomes"):
                    arb_signals = strategy.generate(market_data)
                    signals.extend(arb_signals)
            elif name == "market_making":
                if market_data.get("best_bid") is not None:
                    mm_signals = strategy.generate(market_data)
                    signals.extend(mm_signals)
            else:
                sig = strategy.generate(market_data)
                if sig:
                    signals.append(sig)

        signals.sort(key=lambda s: abs(s.edge), reverse=True)
        return signals

    def should_use_maker(self, market_data: dict) -> bool:

        hours_left = market_data.get("time_to_resolution_hours")
        edge = market_data.get("edge", 0)
        if hours_left is not None and hours_left < 1 and edge > 0.10:
            return False
        return True

    def get_price_zone(self, price: float) -> str:
        if price <= self.price_sweet_spot:
            return "sweet_spot"
        elif price <= self.price_marginal_upper:
            return "marginal"
        elif price <= self.price_danger_zone:
            return "caution"
        else:
            return "high_price"
