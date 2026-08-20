import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import joblib
import numpy as np

from src.utils.fees import polymarket_fee
from src.utils.logger import get_logger

log = get_logger(__name__)

MODELS_DIR = Path(__file__).resolve().parents[2] / "data" / "models"

@dataclass
class Signal:
    token_id: str
    condition_id: str
    side: Literal["YES", "NO"]
    p_model: float
    p_market: float
    edge: float
    ev: float
    source: str
    confidence: float
    market_question: str = ""

class SignalEngine:

    def __init__(
        self,
        tb_model_path: str | None = "lgb_v3.joblib",
        tb_calibrator_path: str | None = "calibrators_v3.joblib",
        tb_meta_path: str | None = "htr_meta_v1.json",
        htr_model_path: str | None = None,
        htr_meta_path: str | None = None,
        fee_rate: float = 0.0175,
        edge_threshold: float = 0.01,
        tb_weight: float = 0.4,
        htr_weight: float = 0.6,
    ):
        self.fee_rate = fee_rate
        self.edge_threshold = edge_threshold
        self.tb_weight = tb_weight
        self.htr_weight = htr_weight

        self.tb_model = None
        self.tb_calibrator = None
        self.tb_features = None
        self.tb_medians = None

        if tb_model_path:
            try:
                self.tb_model = joblib.load(MODELS_DIR / tb_model_path)
                self.tb_calibrator = joblib.load(MODELS_DIR / tb_calibrator_path)
                with open(MODELS_DIR / tb_meta_path) as f:
                    meta = json.load(f)
                self.tb_features = meta["features"]
                self.tb_medians = np.array(meta["train_medians"])
                log.info(f"TB model loaded: {len(self.tb_features)} features")
            except Exception as e:
                log.warning(f"TB model not loaded: {e}")

        self.htr_model = None
        self.htr_calibrator = None
        self.htr_features = None
        self.htr_medians = None

        if htr_model_path:
            try:
                self.htr_model = joblib.load(MODELS_DIR / htr_model_path)
                if htr_meta_path:
                    with open(MODELS_DIR / htr_meta_path) as f:
                        htr_meta = json.load(f)
                    self.htr_features = htr_meta.get("features")
                    self.htr_medians = np.array(htr_meta["train_medians"])

                cal_path = MODELS_DIR / htr_model_path.replace(".joblib", "_platt.joblib")
                if cal_path.exists():
                    self.htr_calibrator = joblib.load(cal_path)
                log.info(f"HTR model loaded: {len(self.htr_features or [])} features")
            except Exception as e:
                log.warning(f"HTR model not loaded: {e}")

    def _fee(self, price: float) -> float:
        return polymarket_fee(price, self.fee_rate)

    def _compute_ev(self, p_model: float, p_market: float, side: str) -> float:
        if side == "YES":
            fee = self._fee(p_market)
            win_net = (1.0 - p_market) - fee
            loss_net = p_market + fee
            ev = p_model * win_net - (1 - p_model) * loss_net
        else:
            no_price = 1.0 - p_market
            fee = self._fee(no_price)
            win_net = p_market - fee
            loss_net = no_price + fee
            ev = (1 - p_model) * win_net - p_model * loss_net
        return ev

    def generate_tb_signal(self, market_data: dict) -> Signal | None:
        if self.tb_model is None:
            return None

        p_market = float(market_data.get("midpoint", 0.5))
        if p_market < 0.25 or p_market > 0.75:
            return None

        features = self._extract_tb_features(market_data)
        if features is None:
            return None

        try:
            X = np.array([features])
            raw_prob = self.tb_model.predict_proba(X)[:, 1][0]
            cal_prob = self.tb_calibrator["platt_lgb"].predict_proba(
                raw_prob.reshape(1, -1)
            )[:, 1][0]

            p_market = float(market_data.get("midpoint", 0.5))
            edge = cal_prob - p_market

            if edge > self.edge_threshold:
                side = "YES"
            elif edge < -self.edge_threshold:
                side = "NO"
            else:
                return None

            ev = self._compute_ev(cal_prob, p_market, side)
            if ev <= 0:
                return None

            return Signal(
                token_id=market_data.get("token_id", ""),
                condition_id=market_data.get("condition_id", ""),
                side=side,
                p_model=cal_prob,
                p_market=p_market,
                edge=edge,
                ev=ev,
                source="tb",
                confidence=min(abs(edge) / 0.10, 1.0),
                market_question=market_data.get("question", ""),
            )
        except Exception as e:
            log.debug(f"TB signal error: {e}")
            return None

    def generate_htr_signal(self, market_data: dict) -> Signal | None:
        if self.htr_model is None:
            return None

        p_market = float(market_data.get("midpoint", 0.5))
        if p_market < 0.02 or p_market > 0.98:
            return None

        features = self._extract_htr_features(market_data)
        if features is None:
            return None

        try:
            X = np.array([features])
            p_yes = self.htr_model.predict_proba(X)[:, 1][0]
            if self.htr_calibrator is not None:
                p_yes = self.htr_calibrator.predict_proba(
                    np.array([[p_yes]])
                )[:, 1][0]
            p_market = float(market_data.get("midpoint", 0.5))
            edge = p_yes - p_market

            if edge > self.edge_threshold:
                side = "YES"
            elif edge < -self.edge_threshold:
                side = "NO"
            else:
                return None

            ev = self._compute_ev(p_yes, p_market, side)
            if ev <= 0:
                return None

            return Signal(
                token_id=market_data.get("token_id", ""),
                condition_id=market_data.get("condition_id", ""),
                side=side,
                p_model=p_yes,
                p_market=p_market,
                edge=edge,
                ev=ev,
                source="htr",
                confidence=min(abs(edge) / 0.10, 1.0),
                market_question=market_data.get("question", ""),
            )
        except Exception as e:
            log.debug(f"HTR signal error: {e}")
            return None

    def generate(self, market_data: dict) -> Signal | None:
        tb_signal = self.generate_tb_signal(market_data)
        htr_signal = self.generate_htr_signal(market_data)

        if tb_signal is None and htr_signal is None:
            return None

        if tb_signal is None:
            return htr_signal

        if htr_signal is None:
            return tb_signal

        if tb_signal.side == htr_signal.side:

            p_ensemble = (
                self.tb_weight * tb_signal.p_model
                + self.htr_weight * htr_signal.p_model
            )
            p_market = tb_signal.p_market
            edge = p_ensemble - p_market
            side = tb_signal.side

            if side == "NO":
                edge = -edge

            ev = self._compute_ev(p_ensemble, p_market, side)
            if ev <= 0:
                return max(tb_signal, htr_signal, key=lambda s: s.ev)

            return Signal(
                token_id=tb_signal.token_id,
                condition_id=tb_signal.condition_id,
                side=side,
                p_model=p_ensemble,
                p_market=p_market,
                edge=edge,
                ev=ev,
                source="ensemble",
                confidence=min(
                    (tb_signal.confidence + htr_signal.confidence) / 1.5, 1.0
                ),
                market_question=tb_signal.market_question,
            )
        else:

            return max(tb_signal, htr_signal, key=lambda s: s.ev)

    def _extract_tb_features(self, market_data: dict) -> np.ndarray | None:
        if self.tb_features is None or self.tb_medians is None:
            return None

        features = self.tb_medians.copy()
        feature_map = {
            "p_price": market_data.get("midpoint"),
            "ob_bestBid": market_data.get("best_bid"),
            "ob_bestAsk": market_data.get("best_ask"),
            "ob_spread": market_data.get("spread"),
            "ob_volumeNum": market_data.get("volume"),
            "ob_liquidityNum": market_data.get("liquidity"),
        }

        for feat_name, value in feature_map.items():
            if value is not None and feat_name in self.tb_features:
                idx = self.tb_features.index(feat_name)
                features[idx] = float(value)

        price = market_data.get("midpoint")
        if price and "p_price_extremity" in self.tb_features:
            idx = self.tb_features.index("p_price_extremity")
            features[idx] = abs(price - 0.5) * 2

        return features

    def _extract_htr_features(self, market_data: dict) -> np.ndarray | None:
        if self.htr_features is None or self.htr_medians is None:
            return None

        price = market_data.get("midpoint")
        if price is None:
            return None

        features = self.htr_medians.copy()

        live_map = {
            "price_mid": price,
            "price_q1": price,
            "price_q3": price,
            "price_std": market_data.get("price_std", 0.0),
            "price_range": market_data.get("price_range", 0.0),
            "mid_extremity": abs(price - 0.5) * 2,
            "q1_to_q3_trend": 0.0,
            "log_volume": np.log1p(market_data.get("volume", 0) or 0),
            "log_lifetime": market_data.get("log_lifetime"),
            "vol_per_hour": market_data.get("vol_per_hour"),
            "neg_risk": float(market_data.get("neg_risk", 0) or 0),
            "spread": market_data.get("spread", 0.0) or 0.0,
            "momentum_first_half": 0.0,
            "momentum_second_half": 0.0,
        }

        for feat_name, value in live_map.items():
            if value is not None and feat_name in self.htr_features:
                idx = self.htr_features.index(feat_name)
                features[idx] = float(value)

        return features
