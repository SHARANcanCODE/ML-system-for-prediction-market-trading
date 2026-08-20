import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

log = get_logger(__name__)

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

def _find_latest(subdir: str, prefix: str, suffix: str = ".json") -> Path | None:
    files = sorted((RAW_DIR / subdir).glob(f"{prefix}*{suffix}"))
    return files[-1] if files else None

def _parse_prices(val) -> list[float] | None:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    if isinstance(val, list):
        return [float(x) for x in val]
    if isinstance(val, str):
        try:
            return [float(x) for x in json.loads(val)]
        except (json.JSONDecodeError, ValueError):
            return None
    return None

def _parse_clob_ids(val) -> list[str]:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return []
    if isinstance(val, list):
        return [str(x) for x in val]
    if isinstance(val, str):
        try:
            return [str(x) for x in json.loads(val)]
        except (json.JSONDecodeError, ValueError):
            return []
    return []

class DataPipeline:

    def __init__(self, raw_dir: Path | str = RAW_DIR, processed_dir: Path | str = PROCESSED_DIR):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def load_markets(self, active: bool = True, resolved: bool = False) -> pd.DataFrame:
        dfs = []

        if active:
            path = _find_latest("markets", "markets_flat_2")
            if path:
                log.info(f"Loading active markets from {path.name}")
                dfs.append(self._parse_markets_file(path))

        if resolved:
            path = _find_latest("markets", "markets_resolved_flat")
            if path:
                log.info(f"Loading resolved markets from {path.name}")
                dfs.append(self._parse_markets_file(path))

        if not dfs:
            log.warning("No market files found")
            return pd.DataFrame()

        df = pd.concat(dfs, ignore_index=True)
        df = df.drop_duplicates(subset="conditionId", keep="last")
        log.info(f"Markets loaded: {len(df)} rows, {len(df.columns)} cols")
        return df

    def _parse_markets_file(self, path: Path) -> pd.DataFrame:
        with open(path) as f:
            data = json.load(f)
        df = pd.DataFrame(data)

        num_cols = [
            "volume", "volumeNum", "liquidity", "liquidityNum", "liquidityClob",
            "volume24hr", "volume1wk", "volume1mo", "volume1yr",
            "volume24hrClob", "volume1wkClob", "volume1moClob", "volume1yrClob",
            "volumeClob", "spread", "bestBid", "bestAsk", "lastTradePrice",
            "oneWeekPriceChange", "oneMonthPriceChange",
            "_event_volume_24hr",
        ]
        for col in num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        bool_cols = ["active", "closed", "negRisk", "acceptingOrders", "_event_neg_risk"]
        for col in bool_cols:
            if col in df.columns:
                df[col] = df[col].astype(bool)

        dt_cols = ["endDate", "startDate", "createdAt", "updatedAt", "endDateIso", "startDateIso"]
        for col in dt_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

        df["_outcome_prices"] = df["outcomePrices"].apply(_parse_prices)
        df["price_yes"] = df["_outcome_prices"].apply(lambda x: x[0] if x and len(x) >= 1 else np.nan)
        df["price_no"] = df["_outcome_prices"].apply(lambda x: x[1] if x and len(x) >= 2 else np.nan)
        df.drop(columns=["_outcome_prices"], inplace=True)

        df["_clob_token_ids"] = df["clobTokenIds"].apply(_parse_clob_ids)
        df["token_id_yes"] = df["_clob_token_ids"].apply(lambda x: x[0] if len(x) >= 1 else None)
        df["token_id_no"] = df["_clob_token_ids"].apply(lambda x: x[1] if len(x) >= 2 else None)
        df.drop(columns=["_clob_token_ids"], inplace=True)

        return df

    def load_events(self, active: bool = True, resolved: bool = False) -> pd.DataFrame:
        dfs = []

        if active:
            path = _find_latest("markets", "events_2")
            if path:
                log.info(f"Loading active events from {path.name}")
                dfs.append(self._parse_events_file(path))

        if resolved:
            path = _find_latest("markets", "events_resolved")
            if path:
                log.info(f"Loading resolved events from {path.name}")
                dfs.append(self._parse_events_file(path))

        if not dfs:
            return pd.DataFrame()

        df = pd.concat(dfs, ignore_index=True)
        df = df.drop_duplicates(subset="id", keep="last")
        log.info(f"Events loaded: {len(df)} rows")
        return df

    def _parse_events_file(self, path: Path) -> pd.DataFrame:
        with open(path) as f:
            data = json.load(f)

        rows = []
        for event in data:
            row = {k: v for k, v in event.items() if k != "markets"}
            row["n_markets"] = len(event.get("markets", []))
            rows.append(row)

        df = pd.DataFrame(rows)

        if "volume24hr" in df.columns:
            df["volume24hr"] = pd.to_numeric(df["volume24hr"], errors="coerce")
        if "liquidity" in df.columns:
            df["liquidity"] = pd.to_numeric(df["liquidity"], errors="coerce")
        if "negRisk" in df.columns:
            df["negRisk"] = df["negRisk"].astype(bool)
        for col in ["createdAt", "updatedAt", "startDate", "endDate"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

        if "tags" in df.columns:
            df["_tags_flat"] = df["tags"].apply(
                lambda x: ",".join(
                    t.get("label", t.get("slug", "")) for t in x
                ) if isinstance(x, list) else ""
            )

        return df

    def load_prices(self, fidelity: int = 60) -> pd.DataFrame:
        rows = []

        json_files = sorted((self.raw_dir / "prices").glob(f"prices_f{fidelity}_2*.json"))
        for path in json_files:
            log.info(f"Loading prices from {path.name}")
            with open(path) as f:
                data = json.load(f)
            for token_id, history in data.items():
                for point in history:
                    rows.append({"token_id": token_id, "timestamp": point["t"], "price": point["p"]})

        jsonl_files = sorted((self.raw_dir / "prices").glob(f"prices_f{fidelity}_incremental_*.jsonl"))
        for path in jsonl_files:
            log.info(f"Loading incremental prices from {path.name}")
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    tid = record.get("token_id", "")
                    for point in record.get("history", []):
                        rows.append({"token_id": tid, "timestamp": point["t"], "price": point["p"]})

        if not rows:
            log.warning(f"No price data found for fidelity={fidelity}")
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        df = df.sort_values(["token_id", "timestamp"]).reset_index(drop=True)
        df = df.drop_duplicates(subset=["token_id", "timestamp"], keep="last")

        log.info(f"Prices loaded: {len(df)} points, {df['token_id'].nunique()} tokens")
        return df

    def load_trades(self) -> pd.DataFrame:
        rows = []

        json_files = sorted((self.raw_dir / "trades").glob("trades_2*.json"))
        for path in json_files:
            log.info(f"Loading trades from {path.name}")
            with open(path) as f:
                data = json.load(f)
            for condition_id, trade_list in data.items():
                for trade in trade_list:
                    trade["conditionId"] = condition_id
                    rows.append(trade)

        jsonl_files = sorted((self.raw_dir / "trades").glob("trades_incremental_*.jsonl"))
        for path in jsonl_files:
            log.info(f"Loading incremental trades from {path.name}")
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rows.append(json.loads(line))

        if not rows:
            log.warning("No trades data found")
            return pd.DataFrame()

        df = pd.DataFrame(rows)

        if "price" in df.columns:
            df["price"] = pd.to_numeric(df["price"], errors="coerce")
        if "size" in df.columns:
            df["size"] = pd.to_numeric(df["size"], errors="coerce")
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)

        id_cols = [c for c in ["id", "transactionHash", "timestamp", "conditionId"] if c in df.columns]
        if id_cols:
            df = df.drop_duplicates(subset=id_cols, keep="last")

        df = df.sort_values(["conditionId", "timestamp"]).reset_index(drop=True)
        log.info(f"Trades loaded: {len(df)} trades, {df['conditionId'].nunique()} markets")
        return df

    def load_snapshots(self) -> pd.DataFrame:
        rows = []

        for pattern_dir, pattern in [
            (self.raw_dir / "snapshots", "orderbooks_*.jsonl"),
            (self.raw_dir / "orderbooks", "orderbooks_*.jsonl"),
        ]:
            files = sorted(pattern_dir.glob(pattern))
            for f in files:
                with open(f) as fh:
                    for line in fh:
                        line = line.strip()
                        if line:
                            rows.append(json.loads(line))

        if not rows:
            log.warning("No snapshot files found")
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        if "ts" in df.columns:
            df["datetime"] = pd.to_datetime(df["ts"], unit="s", utc=True)
        num_cols = ["best_bid", "best_ask", "spread", "bid_depth", "ask_depth",
                    "bid_depth_5", "ask_depth_5", "bid_depth_all", "ask_depth_all"]
        for col in num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.drop_duplicates(subset=["token_id", "ts"], keep="last") if "token_id" in df.columns else df

        log.info(f"Snapshots loaded: {len(df)} rows")
        return df

    def load_resolution_outcomes(self) -> pd.DataFrame:
        markets = self.load_markets(active=False, resolved=True)
        if markets.empty:
            return pd.DataFrame()

        df = markets[markets["closed"] == True].copy()

        df["resolved_yes"] = df["price_yes"] >= 0.99
        df["resolved_no"] = df["price_no"] >= 0.99
        df["is_resolved"] = df["resolved_yes"] | df["resolved_no"]
        df = df[df["is_resolved"]].copy()

        df["outcome"] = df["resolved_yes"].astype(int)

        if "closedTime" in df.columns:
            df["closedTime"] = pd.to_datetime(df["closedTime"], errors="coerce", utc=True)

        keep_cols = [
            "conditionId", "question", "outcome", "lastTradePrice",
            "volumeNum", "liquidityClob", "spread", "negRisk",
            "closedTime", "createdAt", "endDate",
            "_event_id", "_event_title", "_event_neg_risk",
            "token_id_yes", "token_id_no",
        ]
        keep_cols = [c for c in keep_cols if c in df.columns]
        df = df[keep_cols].reset_index(drop=True)

        log.info(f"Resolution outcomes: {len(df)} resolved markets "
                 f"(Yes won: {(df['outcome']==1).sum()}, No won: {(df['outcome']==0).sum()})")
        return df

    def validate(self, df: pd.DataFrame, name: str) -> dict:
        report = {
            "name": name,
            "rows": len(df),
            "cols": len(df.columns),
            "missing": {},
            "duplicates": 0,
            "anomalies": [],
        }

        missing = df.isnull().sum()
        missing = missing[missing > 0]
        report["missing"] = {col: int(cnt) for col, cnt in missing.items()}

        hashable_cols = [c for c in df.columns if df[c].apply(type).isin([list, dict]).sum() == 0]
        report["duplicates"] = int(df[hashable_cols].duplicated().sum()) if hashable_cols else 0

        for col in df.select_dtypes(include=[np.number]).columns:
            series = df[col].dropna()
            if len(series) == 0:
                continue

            if col in ("price", "price_yes", "price_no", "volume", "volumeNum", "size"):
                neg_count = (series < 0).sum()
                if neg_count > 0:
                    report["anomalies"].append(f"{col}: {neg_count} negative values")

            if col in ("price", "price_yes", "price_no", "bestBid", "bestAsk", "lastTradePrice"):
                out_of_range = ((series < 0) | (series > 1)).sum()
                if out_of_range > 0:
                    report["anomalies"].append(f"{col}: {out_of_range} values outside [0,1]")

        log.info(f"Validation [{name}]: {report['rows']} rows, "
                 f"{len(report['missing'])} cols with nulls, "
                 f"{report['duplicates']} duplicates, "
                 f"{len(report['anomalies'])} anomalies")
        for a in report["anomalies"]:
            log.warning(f"  Anomaly: {a}")

        return report

    def merge_markets_events(self, markets: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
        if events.empty or "_event_id" not in markets.columns:
            return markets

        event_cols = ["id"]
        if "_tags_flat" in events.columns:
            event_cols.append("_tags_flat")
        if "tags" in events.columns:
            event_cols.append("tags")

        events_slim = events[event_cols].rename(columns={"id": "_event_id"})
        merged = markets.merge(events_slim, on="_event_id", how="left")
        log.info(f"Merged markets+events: {len(merged)} rows, added {len(event_cols)-1} cols from events")
        return merged

    def save_parquet(self, df: pd.DataFrame, name: str) -> Path:
        path = self.processed_dir / f"{name}.parquet"

        df_save = df.copy()
        for col in df_save.columns:
            if df_save[col].dtype == object:
                sample = df_save[col].dropna().head(1)
                if len(sample) > 0 and isinstance(sample.iloc[0], (list, dict)):
                    df_save[col] = df_save[col].apply(
                        lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (list, dict)) else x
                    )
        df_save.to_parquet(path, index=False)
        size_mb = path.stat().st_size / (1024 * 1024)
        log.info(f"Saved {path} ({size_mb:.1f} MB, {len(df)} rows)")
        return path

    def build_all(self, include_resolved: bool = False) -> dict[str, pd.DataFrame]:
        log.info("=" * 60)
        log.info("Starting full data pipeline")
        log.info("=" * 60)

        result = {}

        markets = self.load_markets(active=True, resolved=include_resolved)
        if not markets.empty:
            self.validate(markets, "markets")

            events = self.load_events(active=True, resolved=include_resolved)
            if not events.empty:
                self.validate(events, "events")
                markets = self.merge_markets_events(markets, events)
                self.save_parquet(events, "events")
                result["events"] = events
            self.save_parquet(markets, "markets")
            result["markets"] = markets

        prices = self.load_prices()
        if not prices.empty:
            self.validate(prices, "prices")
            self.save_parquet(prices, "prices")
            result["prices"] = prices

        trades = self.load_trades()
        if not trades.empty:
            self.validate(trades, "trades")
            self.save_parquet(trades, "trades")
            result["trades"] = trades

        if include_resolved:
            outcomes = self.load_resolution_outcomes()
            if not outcomes.empty:
                self.validate(outcomes, "resolution_outcomes")
                self.save_parquet(outcomes, "resolution_outcomes")
                result["resolution_outcomes"] = outcomes

        snapshots = self.load_snapshots()
        if not snapshots.empty:
            self.validate(snapshots, "snapshots")
            self.save_parquet(snapshots, "snapshots")
            result["snapshots"] = snapshots

        log.info("=" * 60)
        log.info(f"Pipeline complete: {list(result.keys())}")
        log.info("=" * 60)

        return result
