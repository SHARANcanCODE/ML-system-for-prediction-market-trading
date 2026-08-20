"""Tests for risk management module."""

import pytest
from datetime import datetime, timezone, timedelta
from src.risk.manager import RiskManager, RiskConfig, Regime


@pytest.fixture
def rm():
    config = RiskConfig(
        max_position_pct=0.10,
        max_position_usd=500.0,
        max_positions=5,
        max_correlated=2,
        max_category_pct=0.40,
        max_category_positions=3,
        cash_reserve_pct=0.20,
        max_total_exposure_pct=0.80,
        daily_loss_limit_pct=0.05,
        weekly_loss_limit_pct=0.08,
        max_drawdown_pct=0.15,
        drawdown_cut_10_pct=0.10,
        drawdown_cut_15_pct=0.15,
        drawdown_halt_pct=0.20,
        cooldown_loss_threshold=0.03,
        cooldown_minutes=60,
        min_edge=0.01,
        kelly_fraction=0.5,
        min_liquidity=1000.0,
        max_spread=0.05,
        max_high_prob_pct=0.05,
        max_loss_per_resolution=0.10,
        time_exit_hours=72.0,
        edge_exit_threshold=0.005,
    )
    manager = RiskManager(config)
    manager.reset_daily(1000.0)
    manager.update_equity(1000.0)
    return manager


class TestCheckLimits:
    def test_allows_trading_normally(self, rm):
        ok, reason = rm.check_limits(equity=1000.0, n_positions=3)
        assert ok is True

    def test_blocks_at_max_positions(self, rm):
        ok, reason = rm.check_limits(equity=1000.0, n_positions=5)
        assert ok is False
        assert "Max positions" in reason

    def test_daily_loss_limit(self, rm):
        rm.record_trade(-30.0, equity=1000.0)
        rm.record_trade(-25.0, equity=970.0)
        ok, reason = rm.check_limits(equity=945.0, n_positions=0)
        assert ok is False
        assert "Daily loss" in reason
        assert rm.is_halted

    def test_max_drawdown_halt(self, rm):
        rm.update_equity(1000.0)
        ok, reason = rm.check_limits(equity=790.0, n_positions=0)
        assert ok is False
        assert "drawdown" in reason.lower()
        assert rm.is_halted

    def test_halt_persists(self, rm):
        rm.record_trade(-60.0, equity=1000.0)
        rm.check_limits(equity=940.0, n_positions=0)
        assert rm.is_halted
        ok, _ = rm.check_limits(equity=1100.0, n_positions=0)
        assert ok is False

    def test_resume(self, rm):
        rm.record_trade(-60.0, equity=1000.0)
        rm.check_limits(equity=940.0, n_positions=0)
        assert rm.is_halted
        rm.resume()
        assert not rm.is_halted

    def test_cooldown_blocks_trading(self, rm):
        rm._cooldown_until = datetime.now(timezone.utc) + timedelta(minutes=30)
        ok, reason = rm.check_limits(equity=1000.0, n_positions=0)
        assert ok is False
        assert "Cooldown" in reason

    def test_cooldown_expires(self, rm):
        rm._cooldown_until = datetime.now(timezone.utc) - timedelta(minutes=1)
        ok, _ = rm.check_limits(equity=1000.0, n_positions=0)
        assert ok is True


class TestCanOpen:
    def test_basic_open(self, rm):
        ok, _ = rm.can_open(equity=1000, cash=1000, n_positions=0, token_id="tok1")
        assert ok is True

    def test_cash_reserve(self, rm):
        ok, reason = rm.can_open(equity=1000, cash=150, n_positions=0, token_id="tok1")
        assert ok is False
        assert "reserve" in reason.lower()

    def test_total_exposure_limit(self, rm):
        ok, reason = rm.can_open(equity=1000, cash=150, n_positions=0, token_id="tok1")
        assert ok is False

    def test_event_concentration(self, rm):
        rm.position_opened("event1")
        rm.position_opened("event1")
        ok, reason = rm.can_open(
            equity=1000, cash=1000, n_positions=2, token_id="tok3", event_id="event1"
        )
        assert ok is False
        assert "concentration" in reason.lower()

    def test_category_exposure(self, rm):
        rm.position_opened(category="sports", invested=450.0)
        ok, reason = rm.can_open(
            equity=1000, cash=1000, n_positions=1, token_id="tok2", category="sports"
        )
        assert ok is False
        assert "Category exposure" in reason

    def test_category_positions(self, rm):
        for _ in range(3):
            rm.position_opened(category="crypto", invested=50.0)
        ok, reason = rm.can_open(
            equity=1000, cash=1000, n_positions=3, token_id="tok4", category="crypto"
        )
        assert ok is False
        assert "Category positions" in reason

    def test_low_liquidity(self, rm):
        ok, reason = rm.can_open(
            equity=1000, cash=1000, n_positions=0, token_id="tok1",
            market_liquidity=500.0,
        )
        assert ok is False
        assert "liquidity" in reason.lower()

    def test_wide_spread(self, rm):
        ok, reason = rm.can_open(
            equity=1000, cash=1000, n_positions=0, token_id="tok1",
            market_spread=0.08,
        )
        assert ok is False
        assert "spread" in reason.lower()

    def test_weekly_loss_limit(self, rm):
        rm._weekly_pnl = -85.0
        ok, reason = rm.can_open(
            equity=1000, cash=1000, n_positions=0, token_id="tok1"
        )
        assert ok is False
        assert "Weekly" in reason


class TestPositionSize:
    def test_kelly_sizing(self, rm):
        size = rm.compute_size(equity=1000, cash=1000, p_model=0.70, p_market=0.50)
        assert size > 0
        assert size <= 100

    def test_no_edge(self, rm):
        size = rm.compute_size(equity=1000, cash=1000, p_model=0.50, p_market=0.50)
        assert size == 0.0

    def test_negative_edge(self, rm):
        size = rm.compute_size(equity=1000, cash=1000, p_model=0.30, p_market=0.50)
        assert size >= 0

    def test_capped_at_max_usd(self, rm):
        size = rm.compute_size(equity=10000, cash=10000, p_model=0.99, p_market=0.10)
        assert size <= 500

    def test_respects_cash_reserve(self, rm):
        size = rm.compute_size(equity=1000, cash=250, p_model=0.80, p_market=0.40)
        assert size <= 50

    def test_high_prob_bond_cap(self, rm):
        size = rm.compute_size(equity=1000, cash=1000, p_model=0.99, p_market=0.95)
        assert size <= 50

    def test_progressive_drawdown_cut(self, rm):
        normal = rm.compute_size(equity=1000, cash=1000, p_model=0.70, p_market=0.50)
        rm.peak_equity = 1000.0
        rm.daily.starting_equity = 1000.0
        rm.daily.realized_pnl = -120.0
        reduced = rm.compute_size(equity=880, cash=880, p_model=0.70, p_market=0.50)
        assert reduced < normal

    def test_category_cap(self, rm):
        rm.position_opened(category="sports", invested=350.0)
        size = rm.compute_size(
            equity=1000, cash=1000, p_model=0.99, p_market=0.10, category="sports"
        )
        assert size <= 50

    def test_max_position_usd(self, rm):
        size = rm.compute_size(equity=100000, cash=100000, p_model=0.80, p_market=0.40)
        assert size <= 500


class TestExitRules:
    def test_time_exit(self, rm):
        entry = datetime.now(timezone.utc) - timedelta(hours=80)
        should_exit, reason = rm.should_exit(
            regime=Regime.MEAN_REVERTING, entry_time=entry,
            entry_price=0.50, current_price=0.50
        )
        assert should_exit
        assert "Time exit" in reason

    def test_edge_gone_exit(self, rm):
        entry = datetime.now(timezone.utc) - timedelta(hours=1)
        should_exit, reason = rm.should_exit(
            regime=Regime.UNKNOWN, entry_time=entry,
            entry_price=0.50, current_price=0.52,
            p_model_current=0.51, p_market_current=0.51, side="YES"
        )
        assert should_exit
        assert "Edge gone" in reason

    def test_mr_no_stop_loss(self, rm):
        """Chan Ch.6: NO stop loss for mean-reverting!"""
        entry = datetime.now(timezone.utc) - timedelta(hours=5)
        should_exit, reason = rm.should_exit(
            regime=Regime.MEAN_REVERTING, entry_time=entry,
            entry_price=0.50, current_price=0.42, side="YES"
        )
        assert not should_exit
        assert reason == "Hold"

    def test_mr_target_hit(self, rm):
        entry = datetime.now(timezone.utc) - timedelta(hours=5)
        should_exit, reason = rm.should_exit(
            regime=Regime.MEAN_REVERTING, entry_time=entry,
            entry_price=0.50, current_price=0.56, side="YES"
        )
        assert should_exit
        assert "MR target" in reason

    def test_momentum_stop_loss(self, rm):
        """Chan Ch.6: Stop loss OK for momentum."""
        entry = datetime.now(timezone.utc) - timedelta(hours=5)
        should_exit, reason = rm.should_exit(
            regime=Regime.MOMENTUM, entry_time=entry,
            entry_price=0.50, current_price=0.44, side="YES"
        )
        assert should_exit
        assert "Momentum SL" in reason

    def test_momentum_take_profit(self, rm):
        entry = datetime.now(timezone.utc) - timedelta(hours=5)
        should_exit, reason = rm.should_exit(
            regime=Regime.MOMENTUM, entry_time=entry,
            entry_price=0.50, current_price=0.62, side="YES"
        )
        assert should_exit
        assert "Momentum TP" in reason


class TestCooldown:
    def test_big_loss_triggers_cooldown(self, rm):
        rm.record_trade(-40.0, equity=1000.0)
        assert rm.is_cooling_down

    def test_small_loss_no_cooldown(self, rm):
        rm.record_trade(-10.0, equity=1000.0)
        assert not rm.is_cooling_down

    def test_clear_cooldown(self, rm):
        rm.record_trade(-40.0, equity=1000.0)
        assert rm.is_cooling_down
        rm.clear_cooldown()
        assert not rm.is_cooling_down


class TestDrawdownMultiplier:
    def test_no_drawdown(self, rm):
        assert rm.drawdown_size_multiplier == 1.0

    def test_at_10_pct(self, rm):
        rm.peak_equity = 1000.0
        rm.daily.starting_equity = 1000.0
        rm.daily.realized_pnl = -110.0
        assert rm.drawdown_size_multiplier == 0.7

    def test_at_15_pct(self, rm):
        rm.peak_equity = 1000.0
        rm.daily.starting_equity = 1000.0
        rm.daily.realized_pnl = -160.0
        assert rm.drawdown_size_multiplier == 0.5

    def test_at_20_pct(self, rm):
        rm.peak_equity = 1000.0
        rm.daily.starting_equity = 1000.0
        rm.daily.realized_pnl = -210.0
        assert rm.drawdown_size_multiplier == 0.0


class TestEventTracking:
    def test_open_close_tracking(self, rm):
        rm.position_opened("e1")
        rm.position_opened("e1")
        assert rm._event_positions["e1"] == 2
        rm.position_closed("e1", pnl=10.0, equity=1000.0)
        assert rm._event_positions["e1"] == 1
        rm.position_closed("e1", pnl=-5.0, equity=1000.0)
        assert "e1" not in rm._event_positions

    def test_trade_recording(self, rm):
        rm.position_closed("e1", pnl=50.0, equity=1000.0)
        rm.position_closed("e2", pnl=-20.0, equity=1050.0)
        assert rm.daily.n_trades == 2
        assert rm.daily.n_wins == 1
        assert rm.daily.realized_pnl == 30.0

    def test_category_tracking(self, rm):
        rm.position_opened(category="sports", invested=100.0)
        assert rm._category_exposure["sports"] == 100.0
        assert rm._category_positions["sports"] == 1
        rm.position_closed(category="sports", invested=100.0, pnl=10.0, equity=1000.0)
        assert "sports" not in rm._category_exposure


class TestTrailingKelly:
    def test_no_data_uses_default(self, rm):
        assert rm.trailing_kelly_fraction == rm.config.kelly_fraction

    def test_positive_returns(self, rm):
        rm._trailing_returns = [0.01] * 20
        assert rm.trailing_kelly_fraction == rm.config.kelly_fraction

    def test_zero_mean_returns_zero(self, rm):
        rm._trailing_returns = [0.01, -0.01] * 10
        assert rm.trailing_kelly_fraction == 0.0

    def test_negative_mean_returns_zero(self, rm):
        rm._trailing_returns = [-0.01] * 20
        assert rm.trailing_kelly_fraction == 0.0


class TestStatus:
    def test_status_dict(self, rm):
        status = rm.status()
        assert "halted" in status
        assert "daily_pnl" in status
        assert "peak_equity" in status
        assert "drawdown_pct" in status
        assert "trailing_kelly" in status
        assert "cooling_down" in status
        assert "category_exposure" in status
        assert status["halted"] is False
