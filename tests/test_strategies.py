"""Tests for rule-based trading strategies."""

import pytest
from src.strategies.strategies import (
    MeanReversionStrategy, MomentumStrategy,
    ResolutionConvergenceStrategy, ContrarianStrategy,
    NegRiskArbStrategy, MarketMakingStrategy,
    EventDrivenNLPStrategy,
)
from src.strategies.metamodel import StrategyRouter, RouteDecision
from src.risk.manager import Regime


class TestMeanReversion:
    def test_buy_yes_on_low_z(self):
        s = MeanReversionStrategy(z_entry=2.0, min_edge=0.01)
        signal = s.generate({
            "token_id": "t1", "condition_id": "c1",
            "midpoint": 0.40, "z_score": -2.5, "price_mean": 0.55,
            "n_bars": 30,
        })
        assert signal is not None
        assert signal.side == "YES"
        assert signal.regime == Regime.MEAN_REVERTING

    def test_buy_no_on_high_z(self):
        s = MeanReversionStrategy(z_entry=2.0, min_edge=0.01)
        signal = s.generate({
            "token_id": "t1", "condition_id": "c1",
            "midpoint": 0.70, "z_score": 2.5, "price_mean": 0.50,
            "n_bars": 30,
        })
        assert signal is not None
        assert signal.side == "NO"

    def test_no_signal_in_range(self):
        s = MeanReversionStrategy(z_entry=2.0)
        signal = s.generate({
            "token_id": "t1", "condition_id": "c1",
            "midpoint": 0.50, "z_score": 0.5, "price_mean": 0.50,
            "n_bars": 30,
        })
        assert signal is None

    def test_insufficient_bars(self):
        s = MeanReversionStrategy(min_bars=20)
        signal = s.generate({
            "token_id": "t1", "condition_id": "c1",
            "midpoint": 0.40, "z_score": -3.0, "price_mean": 0.55,
            "n_bars": 5,
        })
        assert signal is None


class TestMomentum:
    def test_upward_momentum(self):
        s = MomentumStrategy(min_return=0.05, min_edge=0.01)
        signal = s.generate({
            "token_id": "t1", "condition_id": "c1",
            "midpoint": 0.60, "ret_24h": 0.08, "volume_ratio": 1.5,
        })
        assert signal is not None
        assert signal.side == "YES"
        assert signal.regime == Regime.MOMENTUM

    def test_downward_momentum(self):
        s = MomentumStrategy(min_return=0.05, min_edge=0.01)
        signal = s.generate({
            "token_id": "t1", "condition_id": "c1",
            "midpoint": 0.40, "ret_24h": -0.08, "volume_ratio": 1.5,
        })
        assert signal is not None
        assert signal.side == "NO"

    def test_low_volume_no_signal(self):
        s = MomentumStrategy(min_return=0.05)
        signal = s.generate({
            "token_id": "t1", "condition_id": "c1",
            "midpoint": 0.60, "ret_24h": 0.10, "volume_ratio": 0.8,
        })
        assert signal is None


class TestResolutionConvergence:
    def test_yes_convergence(self):
        s = ResolutionConvergenceStrategy(
            high_prob_threshold=0.90, discount_threshold=0.03
        )
        signal = s.generate({
            "token_id": "t1", "condition_id": "c1",
            "midpoint": 0.85, "p_model": 0.95,
            "time_to_resolution_hours": 24,
        })
        assert signal is not None
        assert signal.side == "YES"
        assert signal.edge > 0.03

    def test_no_convergence(self):
        s = ResolutionConvergenceStrategy(
            high_prob_threshold=0.90, discount_threshold=0.03
        )
        signal = s.generate({
            "token_id": "t1", "condition_id": "c1",
            "midpoint": 0.15, "p_model": 0.05,
            "time_to_resolution_hours": 24,
        })
        assert signal is not None
        assert signal.side == "NO"

    def test_no_discount(self):
        s = ResolutionConvergenceStrategy(discount_threshold=0.03)
        signal = s.generate({
            "token_id": "t1", "condition_id": "c1",
            "midpoint": 0.94, "p_model": 0.95,
        })
        assert signal is None  # Only 1% discount < 3% threshold


class TestContrarian:
    def test_yes_bias_signal(self):
        s = ContrarianStrategy(yes_bias=0.02, min_edge=0.01)
        signal = s.generate({
            "token_id": "t1", "condition_id": "c1",
            "midpoint": 0.55, "calibration_error": 0.05,
        })
        assert signal is not None
        assert signal.side == "NO"
        assert signal.strategy == "contrarian"

    def test_small_bias_no_signal(self):
        s = ContrarianStrategy(yes_bias=0.02)
        signal = s.generate({
            "token_id": "t1", "condition_id": "c1",
            "midpoint": 0.55, "calibration_error": 0.01,
        })
        assert signal is None


class TestNegRiskArb:
    def test_overpriced_sum(self):
        s = NegRiskArbStrategy(min_deviation=0.02)
        signals = s.generate({
            "outcomes": [
                {"token_id": "t1", "condition_id": "c1", "midpoint": 0.40},
                {"token_id": "t2", "condition_id": "c2", "midpoint": 0.35},
                {"token_id": "t3", "condition_id": "c3", "midpoint": 0.35},
            ]
        })
        # Sum = 1.10, deviation = +0.10 → should generate NO signals
        assert len(signals) > 0
        assert all(s.side == "NO" for s in signals)
        assert all(s.strategy == "negrisk_arb" for s in signals)

    def test_underpriced_sum(self):
        s = NegRiskArbStrategy(min_deviation=0.02)
        signals = s.generate({
            "outcomes": [
                {"token_id": "t1", "condition_id": "c1", "midpoint": 0.25},
                {"token_id": "t2", "condition_id": "c2", "midpoint": 0.25},
                {"token_id": "t3", "condition_id": "c3", "midpoint": 0.20},
            ]
        })
        # Sum = 0.70 → buy YES
        assert len(signals) > 0
        assert all(s.side == "YES" for s in signals)

    def test_no_arb_opportunity(self):
        s = NegRiskArbStrategy(min_deviation=0.02)
        signals = s.generate({
            "outcomes": [
                {"token_id": "t1", "condition_id": "c1", "midpoint": 0.50},
                {"token_id": "t2", "condition_id": "c2", "midpoint": 0.50},
            ]
        })
        assert len(signals) == 0


class TestMarketMaking:
    def _base_data(self, **overrides):
        d = {
            "token_id": "t1", "condition_id": "c1",
            "best_bid": 0.40, "best_ask": 0.46, "midpoint": 0.43,
            "inventory_yes": 0.0, "inventory_no": 0.0,
            "volatility": 0.01,
        }
        d.update(overrides)
        return d

    def test_generates_bid_and_ask(self):
        s = MarketMakingStrategy(min_spread=0.02, min_edge=0.001)
        signals = s.generate(self._base_data())
        assert len(signals) == 2
        sides = {sig.side for sig in signals}
        assert sides == {"YES", "NO"}
        assert all(sig.strategy == "market_making" for sig in signals)

    def test_tight_spread_no_signal(self):
        s = MarketMakingStrategy(min_spread=0.02)
        signals = s.generate(self._base_data(best_bid=0.44, best_ask=0.45))
        assert len(signals) == 0

    def test_inventory_skew(self):
        s = MarketMakingStrategy(min_spread=0.02, min_edge=0.001)
        # Heavy YES inventory → skew quotes down (lower bid, lower ask)
        signals_heavy = s.generate(self._base_data(inventory_yes=400.0))
        signals_neutral = s.generate(self._base_data())
        bid_heavy = [s for s in signals_heavy if s.meta["quote_type"] == "bid"][0]
        bid_neutral = [s for s in signals_neutral if s.meta["quote_type"] == "bid"][0]
        assert bid_heavy.meta["our_bid"] < bid_neutral.meta["our_bid"]

    def test_inventory_cap_blocks_side(self):
        s = MarketMakingStrategy(min_spread=0.02, min_edge=0.001, max_inventory=100)
        signals = s.generate(self._base_data(inventory_yes=200.0))
        # YES side capped, only NO signal
        sides = [sig.side for sig in signals]
        assert "YES" not in sides
        assert "NO" in sides

    def test_volatility_widens_quotes(self):
        s = MarketMakingStrategy(min_spread=0.02, min_edge=0.001)
        signals_calm = s.generate(self._base_data(volatility=0.0))
        signals_vol = s.generate(self._base_data(volatility=0.05))
        bid_calm = [s for s in signals_calm if s.meta["quote_type"] == "bid"][0]
        bid_vol = [s for s in signals_vol if s.meta["quote_type"] == "bid"][0]
        # Higher vol → wider quotes → lower bid
        assert bid_vol.meta["our_bid"] < bid_calm.meta["our_bid"]

    def test_missing_data_no_signal(self):
        s = MarketMakingStrategy()
        assert s.generate({"token_id": "t1"}) == []


class TestStrategyRouter:
    def test_negrisk_highest_priority(self):
        router = StrategyRouter()
        decision = router.route({
            "midpoint": 0.50, "liquidity": 5000,
            "is_negrisk": True,
            "outcomes": [
                {"token_id": "t1", "condition_id": "c1", "midpoint": 0.40},
                {"token_id": "t2", "condition_id": "c2", "midpoint": 0.35},
                {"token_id": "t3", "condition_id": "c3", "midpoint": 0.35},
            ],
        })
        assert decision.strategy_name == "negrisk_arb"
        assert decision.priority == 1
        assert not decision.skip

    def test_convergence_second_priority(self):
        router = StrategyRouter()
        decision = router.route({
            "midpoint": 0.85, "liquidity": 5000,
            "p_model": 0.95,
            "time_to_resolution_hours": 24,
        })
        assert decision.strategy_name == "convergence"
        assert decision.priority == 2

    def test_low_liquidity_skipped(self):
        router = StrategyRouter()
        decision = router.route({
            "midpoint": 0.50, "liquidity": 100,
        })
        assert decision.skip
        assert "liquidity" in decision.skip_reason.lower()

    def test_culture_high_price_skipped(self):
        router = StrategyRouter()
        decision = router.route({
            "midpoint": 0.85, "liquidity": 5000,
            "cat_is_culture": 1,
        })
        assert decision.skip
        assert "culture" in decision.skip_reason.lower()

    def test_mr_at_low_price_priority_3(self):
        router = StrategyRouter()
        decision = router.route({
            "midpoint": 0.15, "liquidity": 5000,
            "z_score": -2.5, "price_mean": 0.30, "n_bars": 30,
            "token_id": "t1", "condition_id": "c1",
        })
        assert decision.strategy_name == "mean_reversion"
        assert decision.priority == 3

    def test_nlp_strategy_in_router(self):
        router = StrategyRouter()
        decision = router.route({
            "midpoint": 0.40, "liquidity": 5000,
            "nlp_comment_sentiment": -0.30,
            "nlp_comment_count": 10,
            "nlp_comment_velocity": 5.0,
            "nlp_bullish_keyword_ratio": 0.3,
            "token_id": "t1", "condition_id": "c1",
        })
        assert decision.strategy_name == "event_driven_nlp"
        assert decision.priority == 5
        assert not decision.skip

    def test_market_making_in_router(self):
        router = StrategyRouter()
        decision = router.route({
            "midpoint": 0.50, "liquidity": 5000,
            "best_bid": 0.47, "best_ask": 0.53,
            "token_id": "t1", "condition_id": "c1",
        })
        assert decision.strategy_name == "market_making"
        assert decision.priority == 8
        assert not decision.skip

    def test_no_signal_returns_skip(self):
        router = StrategyRouter()
        decision = router.route({
            "midpoint": 0.50, "liquidity": 5000,
        })
        assert decision.skip

    def test_always_maker_by_default(self):
        router = StrategyRouter()
        assert router.should_use_maker({"midpoint": 0.50})

    def test_taker_for_urgent_high_edge(self):
        router = StrategyRouter()
        assert not router.should_use_maker({
            "time_to_resolution_hours": 0.5, "edge": 0.15,
        })

    def test_price_zone_classification(self):
        router = StrategyRouter()
        assert router.get_price_zone(0.10) == "sweet_spot"
        assert router.get_price_zone(0.35) == "marginal"
        assert router.get_price_zone(0.60) == "caution"
        assert router.get_price_zone(0.85) == "high_price"

    def test_generate_signals_ranks_by_edge(self):
        router = StrategyRouter()
        signals = router.generate_signals({
            "token_id": "t1", "condition_id": "c1",
            "midpoint": 0.40, "z_score": -2.5, "price_mean": 0.55,
            "n_bars": 30, "ret_24h": -0.08, "volume_ratio": 1.5,
        })
        if len(signals) >= 2:
            assert abs(signals[0].edge) >= abs(signals[1].edge)


class TestEventDrivenNLP:
    """Tests for EventDrivenNLPStrategy (contrarian sentiment)."""

    def _base_data(self, **overrides):
        data = {
            "token_id": "t1", "condition_id": "c1",
            "midpoint": 0.40,
            "nlp_comment_sentiment": -0.30,
            "nlp_comment_count": 10,
            "nlp_comment_velocity": 5.0,
            "nlp_bullish_keyword_ratio": 0.3,
        }
        data.update(overrides)
        return data

    def test_negative_sentiment_buys_yes(self):
        s = EventDrivenNLPStrategy(min_edge=0.01)
        signal = s.generate(self._base_data(nlp_comment_sentiment=-0.30))
        assert signal is not None
        assert signal.side == "YES"
        assert signal.strategy == "event_driven_nlp"
        assert signal.regime == Regime.MEAN_REVERTING

    def test_positive_sentiment_high_price_buys_no(self):
        s = EventDrivenNLPStrategy(min_edge=0.001)
        signal = s.generate(self._base_data(
            nlp_comment_sentiment=0.40,
            midpoint=0.75,
        ))
        assert signal is not None
        assert signal.side == "NO"

    def test_neutral_sentiment_no_signal(self):
        s = EventDrivenNLPStrategy()
        signal = s.generate(self._base_data(nlp_comment_sentiment=-0.05))
        assert signal is None

    def test_too_few_comments_no_signal(self):
        s = EventDrivenNLPStrategy(min_comments=5)
        signal = s.generate(self._base_data(nlp_comment_count=2))
        assert signal is None

    def test_missing_sentiment_no_signal(self):
        s = EventDrivenNLPStrategy()
        signal = s.generate({"token_id": "t1", "midpoint": 0.50})
        assert signal is None

    def test_bearish_keywords_boost_edge(self):
        s = EventDrivenNLPStrategy(min_edge=0.001)
        sig_bearish = s.generate(self._base_data(nlp_bullish_keyword_ratio=0.2))
        sig_neutral = s.generate(self._base_data(nlp_bullish_keyword_ratio=0.6))
        assert sig_bearish is not None
        assert sig_neutral is not None
        assert sig_bearish.edge > sig_neutral.edge

    def test_velocity_boosts_confidence(self):
        s = EventDrivenNLPStrategy(min_edge=0.001)
        sig_fast = s.generate(self._base_data(nlp_comment_velocity=20.0))
        sig_slow = s.generate(self._base_data(nlp_comment_velocity=0.5))
        assert sig_fast is not None
        assert sig_slow is not None
        assert sig_fast.confidence > sig_slow.confidence

    def test_edge_too_small_after_fees(self):
        s = EventDrivenNLPStrategy(min_edge=0.05)
        # Weak sentiment → small edge → filtered by fees
        signal = s.generate(self._base_data(nlp_comment_sentiment=-0.16))
        assert signal is None

    def test_meta_contains_nlp_data(self):
        s = EventDrivenNLPStrategy(min_edge=0.001)
        signal = s.generate(self._base_data())
        assert signal is not None
        assert "sentiment" in signal.meta
        assert "n_comments" in signal.meta
        assert "velocity" in signal.meta
        assert signal.meta["signal_type"] == "contrarian_sentiment"
