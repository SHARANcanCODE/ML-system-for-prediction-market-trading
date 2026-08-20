"""Tests for paper trading engine."""

import pytest
from src.execution.paper_trader import (
    PaperPortfolio, Position, Trade, fee
)


class TestFee:
    def test_fee_midpoint(self):
        # fee(0.5) = 0.5 * 0.5 * 0.0175 = 0.004375
        assert abs(fee(0.5) - 0.004375) < 1e-6

    def test_fee_extreme_low(self):
        # fee(0.01) = 0.01 * 0.99 * 0.0175
        assert fee(0.01) == pytest.approx(0.01 * 0.99 * 0.0175, rel=1e-4)

    def test_fee_extreme_high(self):
        assert fee(0.99) == pytest.approx(0.99 * 0.01 * 0.0175, rel=1e-4)

    def test_fee_clipped(self):
        # Out of range values should be clipped
        assert fee(0.0) > 0  # clipped to 0.01
        assert fee(1.0) > 0  # clipped to 0.99

    def test_fee_symmetric(self):
        # fee(p) should roughly equal fee(1-p) for maker rate
        assert abs(fee(0.3) - fee(0.7)) < 1e-6

    def test_custom_rate(self):
        # Taker rate 10%
        f = fee(0.5, rate=0.10)
        assert f == pytest.approx(0.5 * 0.5 * 0.10)


class TestPaperPortfolio:
    def test_initial_state(self):
        p = PaperPortfolio(initial_capital=500.0, cash=500.0)
        assert p.equity == 500.0
        assert p.n_positions == 0
        assert p.total_pnl == 0.0
        assert p.win_rate == 0.0

    def test_equity_with_positions(self):
        p = PaperPortfolio(initial_capital=1000.0, cash=800.0)
        p.positions["tok1"] = Position(
            token_id="tok1", condition_id="c1", side="YES",
            entry_price=0.40, n_shares=500, entry_fee=3.0,
            entry_time="2026-01-01T00:00:00Z",
        )
        # invested = 0.40 * 500 = 200
        assert p.total_invested == 200.0
        assert p.equity == 1000.0  # 800 cash + 200 invested

    def test_win_rate(self):
        p = PaperPortfolio()
        p.trades.append(Trade(
            token_id="t1", condition_id="c1", side="YES",
            entry_price=0.40, exit_price=0.50, n_shares=100,
            entry_fee=1, exit_fee=1, pnl=8.0,
            entry_time="", exit_time="", exit_reason="TP",
        ))
        p.trades.append(Trade(
            token_id="t2", condition_id="c2", side="YES",
            entry_price=0.40, exit_price=0.30, n_shares=100,
            entry_fee=1, exit_fee=1, pnl=-12.0,
            entry_time="", exit_time="", exit_reason="SL",
        ))
        assert p.win_rate == 0.5
        assert p.total_pnl == -4.0

    def test_snapshot(self):
        p = PaperPortfolio(initial_capital=1000.0, cash=1000.0)
        snap = p.snapshot()
        assert "time" in snap
        assert snap["cash"] == 1000.0
        assert snap["n_positions"] == 0
        assert snap["equity"] == 1000.0
