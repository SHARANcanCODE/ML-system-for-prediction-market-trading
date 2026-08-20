from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Literal

import numpy as np

from src.utils.logger import get_logger

log = get_logger(__name__)

class Regime(Enum):
    MEAN_REVERTING = "mr"
    MOMENTUM = "momentum"
    UNKNOWN = "unknown"

@dataclass
class RiskConfig:

    max_position_pct: float = 0.10
    max_position_usd: float = 500.0
    max_positions: int = 10
    max_correlated: int = 3

    max_category_pct: float = 0.40
    max_category_positions: int = 5

    cash_reserve_pct: float = 0.20
    max_total_exposure_pct: float = 0.80

    daily_loss_limit_pct: float = 0.05
    weekly_loss_limit_pct: float = 0.08
    max_drawdown_pct: float = 0.15

    drawdown_cut_10_pct: float = 0.10
    drawdown_cut_15_pct: float = 0.15
    drawdown_halt_pct: float = 0.20

    cooldown_loss_threshold: float = 0.03
    cooldown_minutes: int = 60

    min_edge: float = 0.01
    kelly_fraction: float = 0.5

    min_liquidity: float = 1000.0
    max_spread: float = 0.05

    max_high_prob_pct: float = 0.05
    max_loss_per_resolution: float = 0.10

    time_exit_hours: float = 72.0
    edge_exit_threshold: float = 0.005

@dataclass
class DailyStats:
    date: str = ""
    starting_equity: float = 0.0
    realized_pnl: float = 0.0
    n_trades: int = 0
    n_wins: int = 0
    peak_equity: float = 0.0

    @property
    def daily_return(self) -> float:
        if self.starting_equity <= 0:
            return 0.0
        return self.realized_pnl / self.starting_equity

    @property
    def win_rate(self) -> float:
        if self.n_trades == 0:
            return 0.0
        return self.n_wins / self.n_trades

class RiskManager:

    def __init__(self, config: RiskConfig | None = None):
        self.config = config or RiskConfig()
        self.daily = DailyStats()
        self.peak_equity = 0.0
        self._halted = False
        self._halt_reason = ""
        self._event_positions: dict[str, int] = {}
        self._category_exposure: dict[str, float] = {}
        self._category_positions: dict[str, int] = {}
        self._weekly_pnl: float = 0.0
        self._week_start: str = ""
        self._cooldown_until: datetime | None = None
        self._trailing_returns: list[float] = []

    def reset_daily(self, equity: float):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self.daily.date != today:

            if self._halted:
                log.info(f"Daily reset clears halt: {self._halt_reason}")
                self._halted = False
                self._halt_reason = ""
            self.daily = DailyStats(
                date=today,
                starting_equity=equity,
                peak_equity=equity,
            )
            self.peak_equity = max(self.peak_equity, equity)

            week = datetime.now(timezone.utc).strftime("%Y-W%W")
            if self._week_start != week:
                self._weekly_pnl = 0.0
                self._week_start = week
            log.info(f"Daily reset: equity=${equity:.2f}")

    def update_equity(self, equity: float):
        self.peak_equity = max(self.peak_equity, equity)
        self.daily.peak_equity = max(self.daily.peak_equity, equity)

    def record_trade(self, pnl: float, equity: float = 0.0):
        self.daily.realized_pnl += pnl
        self.daily.n_trades += 1
        self._weekly_pnl += pnl
        if pnl > 0:
            self.daily.n_wins += 1

        if equity > 0:
            self._trailing_returns.append(pnl / equity)

            if len(self._trailing_returns) > 126:
                self._trailing_returns = self._trailing_returns[-126:]

        if equity > 0 and pnl < 0 and abs(pnl) / equity >= self.config.cooldown_loss_threshold:
            self._cooldown_until = datetime.now(timezone.utc) + timedelta(
                minutes=self.config.cooldown_minutes
            )
            log.warning(
                f"Cooldown activated: loss {pnl/equity:.1%} > "
                f"{self.config.cooldown_loss_threshold:.1%}, "
                f"pausing {self.config.cooldown_minutes}min"
            )

    @property
    def is_halted(self) -> bool:
        return self._halted

    @property
    def halt_reason(self) -> str:
        return self._halt_reason

    @property
    def is_cooling_down(self) -> bool:
        if self._cooldown_until is None:
            return False
        if datetime.now(timezone.utc) >= self._cooldown_until:
            self._cooldown_until = None
            return False
        return True

    @property
    def drawdown_pct(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        current = self.daily.starting_equity + self.daily.realized_pnl
        return max(0.0, (self.peak_equity - current) / self.peak_equity)

    @property
    def drawdown_size_multiplier(self) -> float:
        dd = self.drawdown_pct
        if dd >= self.config.drawdown_halt_pct:
            return 0.0
        if dd >= self.config.drawdown_cut_15_pct:
            return 0.5
        if dd >= self.config.drawdown_cut_10_pct:
            return 0.7
        return 1.0

    @property
    def trailing_kelly_fraction(self) -> float:
        if len(self._trailing_returns) < 10:
            return self.config.kelly_fraction
        rets = np.array(self._trailing_returns)
        m = rets.mean()
        s2 = rets.var()
        if s2 <= 0 or m <= 0:
            return 0.0
        raw_kelly = m / s2

        return min(raw_kelly * 0.5, self.config.kelly_fraction)

    def check_limits(self, equity: float, n_positions: int) -> tuple[bool, str]:
        if self._halted:
            return False, f"HALTED: {self._halt_reason}"

        if self.daily.starting_equity > 0:
            daily_loss = -self.daily.realized_pnl / self.daily.starting_equity
            if daily_loss >= self.config.daily_loss_limit_pct:
                self._halted = True
                self._halt_reason = (
                    f"Daily loss limit: {daily_loss:.1%} >= "
                    f"{self.config.daily_loss_limit_pct:.1%}"
                )
                log.warning(f"HALT: {self._halt_reason}")
                return False, self._halt_reason

        if self.is_cooling_down:
            remaining = (self._cooldown_until - datetime.now(timezone.utc)).total_seconds() / 60
            return False, f"Cooldown: {remaining:.0f}min remaining"

        if self.peak_equity > 0:
            drawdown = (self.peak_equity - equity) / self.peak_equity
            if drawdown >= self.config.drawdown_halt_pct:
                self._halted = True
                self._halt_reason = (
                    f"Max drawdown halt: {drawdown:.1%} >= "
                    f"{self.config.drawdown_halt_pct:.1%}"
                )
                log.warning(f"HALT: {self._halt_reason}")
                return False, self._halt_reason

        if n_positions >= self.config.max_positions:
            return False, f"Max positions reached: {n_positions}"

        return True, "OK"

    def can_open(
        self,
        equity: float,
        cash: float,
        n_positions: int,
        token_id: str,
        event_id: str = "",
        category: str = "",
        market_liquidity: float = 0.0,
        market_spread: float = 0.0,
        market_price: float = 0.0,
    ) -> tuple[bool, str]:

        ok, reason = self.check_limits(equity, n_positions)
        if not ok:
            return False, reason

        available = cash - (equity * self.config.cash_reserve_pct)
        if available <= 0:
            return False, f"Cash reserve: ${cash:.0f} < {self.config.cash_reserve_pct:.0%} of ${equity:.0f}"

        total_invested = equity - cash
        if equity > 0 and total_invested / equity >= self.config.max_total_exposure_pct:
            return False, (
                f"Total exposure: {total_invested/equity:.0%} >= "
                f"{self.config.max_total_exposure_pct:.0%}"
            )

        if event_id and self._event_positions.get(event_id, 0) >= self.config.max_correlated:
            return False, f"Event concentration: {self._event_positions[event_id]} positions in event"

        if category:
            cat_exposure = self._category_exposure.get(category, 0.0)
            if equity > 0 and cat_exposure / equity >= self.config.max_category_pct:
                return False, (
                    f"Category exposure: {category} at "
                    f"{cat_exposure/equity:.0%} >= {self.config.max_category_pct:.0%}"
                )
            cat_count = self._category_positions.get(category, 0)
            if cat_count >= self.config.max_category_positions:
                return False, f"Category positions: {category} has {cat_count} positions"

        if market_price >= 0.90 or market_price <= 0.10:

            pass

        if market_liquidity > 0 and market_liquidity < self.config.min_liquidity:
            return False, f"Low liquidity: ${market_liquidity:.0f} < ${self.config.min_liquidity:.0f}"

        if market_spread > 0 and market_spread > self.config.max_spread:
            return False, f"Wide spread: {market_spread:.3f} > {self.config.max_spread:.3f}"

        if self.daily.starting_equity > 0 and self._weekly_pnl < 0:
            weekly_loss = abs(self._weekly_pnl) / self.daily.starting_equity
            if weekly_loss >= self.config.weekly_loss_limit_pct:
                return False, f"Weekly loss limit: {weekly_loss:.1%} >= {self.config.weekly_loss_limit_pct:.1%}"

        return True, "OK"

    def position_opened(self, event_id: str = "", category: str = "",
                        invested: float = 0.0):
        if event_id:
            self._event_positions[event_id] = self._event_positions.get(event_id, 0) + 1
        if category:
            self._category_exposure[category] = self._category_exposure.get(category, 0.0) + invested
            self._category_positions[category] = self._category_positions.get(category, 0) + 1

    def position_closed(self, event_id: str = "", category: str = "",
                        pnl: float = 0.0, invested: float = 0.0,
                        equity: float = 0.0):
        if event_id and event_id in self._event_positions:
            self._event_positions[event_id] = max(0, self._event_positions[event_id] - 1)
            if self._event_positions[event_id] == 0:
                del self._event_positions[event_id]
        if category and category in self._category_exposure:
            self._category_exposure[category] = max(0.0, self._category_exposure[category] - invested)
            if self._category_exposure[category] <= 0:
                del self._category_exposure[category]
            self._category_positions[category] = max(0, self._category_positions.get(category, 1) - 1)
            if self._category_positions.get(category, 0) <= 0:
                self._category_positions.pop(category, None)
        self.record_trade(pnl, equity)

    def should_exit(
        self,
        regime: Regime,
        entry_time: datetime,
        entry_price: float,
        current_price: float,
        p_model_current: float = 0.0,
        p_market_current: float = 0.0,
        side: str = "YES",
    ) -> tuple[bool, str]:
        now = datetime.now(timezone.utc)
        hold_hours = (now - entry_time).total_seconds() / 3600

        if hold_hours >= self.config.time_exit_hours:
            return True, f"Time exit: held {hold_hours:.0f}h >= {self.config.time_exit_hours:.0f}h"

        if p_model_current > 0 and p_market_current > 0:
            if side == "YES":
                current_edge = p_model_current - p_market_current
            else:
                current_edge = (1 - p_model_current) - (1 - p_market_current)
            if current_edge < self.config.edge_exit_threshold:
                return True, f"Edge gone: {current_edge:.4f} < {self.config.edge_exit_threshold}"

        move = current_price - entry_price

        if regime == Regime.MOMENTUM:

            if move <= -0.05:
                return True, f"Momentum SL: move {move:+.3f}"

            if move >= 0.10:
                return True, f"Momentum TP: move {move:+.3f}"

        elif regime == Regime.MEAN_REVERTING:

            if move >= 0.05:
                return True, f"MR target hit: move {move:+.3f}"

        else:

            if move <= -0.10:
                return True, f"Wide SL: move {move:+.3f}"
            if move >= 0.08:
                return True, f"TP: move {move:+.3f}"

        if self.daily.starting_equity > 0:
            daily_loss_pct = -self.daily.realized_pnl / self.daily.starting_equity
            if daily_loss_pct >= self.config.daily_loss_limit_pct:
                return True, f"Portfolio daily stop: {daily_loss_pct:.1%} loss"

        return False, "Hold"

    def compute_size(
        self,
        equity: float,
        cash: float,
        p_model: float,
        p_market: float,
        fee_rate: float = 0.0175,
        category: str = "",
    ) -> float:

        fee = p_market * (1 - p_market) * fee_rate

        if p_model > p_market:
            win_payoff = (1.0 - p_market) - fee
            loss_payoff = p_market + fee
        else:
            no_price = 1.0 - p_market
            fee_no = no_price * (1 - no_price) * fee_rate
            win_payoff = p_market - fee_no
            loss_payoff = no_price + fee_no
            p_model = 1.0 - p_model

        if win_payoff <= 0 or loss_payoff <= 0:
            return 0.0

        b = win_payoff / loss_payoff
        kelly_f = (p_model * b - (1 - p_model)) / b
        kelly_f = max(0.0, kelly_f)

        fraction = self.trailing_kelly_fraction
        f = kelly_f * fraction

        f = min(f, self.config.max_position_pct)

        f *= self.drawdown_size_multiplier
        if f <= 0:
            return 0.0

        if p_market >= 0.90 or p_market <= 0.10:
            f = min(f, self.config.max_high_prob_pct)

        potential_loss_pct = f
        if potential_loss_pct > self.config.max_loss_per_resolution:
            f = self.config.max_loss_per_resolution

        available = cash - (equity * self.config.cash_reserve_pct)
        if available <= 0:
            return 0.0

        if category and equity > 0:
            cat_current = self._category_exposure.get(category, 0.0)
            cat_max = equity * self.config.max_category_pct
            cat_available = max(0.0, cat_max - cat_current)
            available = min(available, cat_available)

        size = min(f * equity, available, self.config.max_position_usd)
        return max(0.0, size)

    def resume(self):
        if self._halted:
            log.info(f"Resuming trading (was halted: {self._halt_reason})")
            self._halted = False
            self._halt_reason = ""

    def clear_cooldown(self):
        self._cooldown_until = None

    def status(self) -> dict:
        return {
            "halted": self._halted,
            "halt_reason": self._halt_reason,
            "daily_pnl": self.daily.realized_pnl,
            "daily_return": self.daily.daily_return,
            "daily_trades": self.daily.n_trades,
            "daily_wr": self.daily.win_rate,
            "weekly_pnl": self._weekly_pnl,
            "peak_equity": self.peak_equity,
            "drawdown_pct": self.drawdown_pct,
            "drawdown_multiplier": self.drawdown_size_multiplier,
            "trailing_kelly": self.trailing_kelly_fraction,
            "cooling_down": self.is_cooling_down,
            "event_positions": dict(self._event_positions),
            "category_exposure": dict(self._category_exposure),
            "category_positions": dict(self._category_positions),
        }
