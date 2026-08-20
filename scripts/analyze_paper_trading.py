import json
import os
import re
from collections import defaultdict
from datetime import datetime, timedelta

DATA_DIR = "logs/paper_trading_vps/paper_trading"

CATEGORY_KEYWORDS = {
    "sports": [
        "nba", "nfl", "nhl", "mlb", "mls", "epl", "premier league",
        "la liga", "serie a", "bundesliga", "ligue 1", "champions league",
        "europa league", "soccer", "football", "basketball", "baseball",
        "hockey", "tennis", "golf", "boxing", "mma", "ufc", "f1",
        "formula 1", "nascar", "cricket", "rugby", "olympics",
        "world cup", "super bowl", "march madness", "playoffs",
        "win on 2026", "spread", "o/u", "over/under", "total points",
        "total goals", "score", "match winner", "vs.", "game",
        "cavaliers", "celtics", "lakers", "warriors", "76ers", "nets",
        "grizzlies", "pistons", "clippers", "rockets", "spurs", "heat",
        "knicks", "bucks", "nuggets", "suns", "mavericks", "thunder",
        "timberwolves", "pacers", "hawks", "bulls", "magic", "hornets",
        "wizards", "raptors", "blazers", "kings", "pelicans", "jazz",
        "lpl", "lol", "esports", "bo5", "bo3",
    ],
    "politics": [
        "trump", "biden", "president", "election", "congress", "senate",
        "governor", "mayor", "democrat", "republican", "gop", "potus",
        "cabinet", "impeach", "legislation", "vote", "poll",
        "parliamentary", "minister", "prime minister",
    ],
    "crypto": [
        "bitcoin", "btc", "ethereum", "eth", "solana", "sol",
        "crypto", "token price", "defi", "nft", "altcoin",
        "dogecoin", "xrp", "cardano", "binance",
    ],
    "commodity": [
        "crude oil", "wti", "brent", "natural gas", "gold price",
        "silver price", "oil price", "commodity", "wheat", "corn",
    ],
    "economics": [
        "fed", "federal reserve", "interest rate", "inflation",
        "gdp", "unemployment", "jobs report", "cpi", "ppi",
        "tariff", "recession", "rate cut", "rate hike",
    ],
    "geopolitics": [
        "iran", "russia", "ukraine", "china", "war", "sanctions",
        "nato", "ceasefire", "invasion", "military", "nuclear",
        "supreme leader", "khamenei", "us forces",
    ],
}

def classify_category(question: str) -> str:
    q = question.lower()
    for cat, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in q:
                return cat
    return "other"

def load_instance(path: str, name: str) -> dict:
    cp_path = os.path.join(path, "checkpoint.json")
    if not os.path.exists(cp_path):
        return None

    with open(cp_path) as f:
        cp = json.load(f)

    trades = cp.get("trades", [])
    positions = cp.get("positions", [])
    cash = cp.get("cash", 0)
    initial = cp.get("initial_capital", 1000)

    for t in trades:
        t["cat_fixed"] = classify_category(t.get("market_question", ""))

    equity_files = sorted(
        [f for f in os.listdir(path) if f.startswith("paper_equity_") and f.endswith(".csv")],
        key=lambda x: os.path.getmtime(os.path.join(path, x)),
        reverse=True,
    )

    last_equity = None
    mtm_equity = None
    if equity_files:
        with open(os.path.join(path, equity_files[0])) as f:
            lines = f.readlines()
            if len(lines) > 1:
                last_line = lines[-1].strip().split(",")

                if len(last_line) >= 9:
                    mtm_equity = float(last_line[8]) if last_line[8] else None

    return {
        "name": name,
        "trades": trades,
        "positions": positions,
        "cash": cash,
        "initial": initial,
        "mtm_equity": mtm_equity,
        "n_trades": len(trades),
    }

def analyze_trades(trades: list) -> dict:
    if not trades:
        return {"n": 0}

    total_pnl = sum(t["pnl"] for t in trades)
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    wr = len(wins) / len(trades) * 100 if trades else 0

    avg_win = sum(t["pnl"] for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t["pnl"] for t in losses) / len(losses) if losses else 0
    pf = abs(sum(t["pnl"] for t in wins) / sum(t["pnl"] for t in losses)) if losses and sum(t["pnl"] for t in losses) != 0 else float("inf")

    cum_pnl = 0
    peak = 0
    max_dd = 0
    for t in sorted(trades, key=lambda x: x.get("exit_time", "")):
        cum_pnl += t["pnl"]
        peak = max(peak, cum_pnl)
        dd = peak - cum_pnl
        max_dd = max(max_dd, dd)

    total_fees = sum(t.get("entry_fee", 0) + t.get("exit_fee", 0) for t in trades)
    gross_pnl = total_pnl + total_fees

    return {
        "n": len(trades),
        "pnl": total_pnl,
        "gross_pnl": gross_pnl,
        "fees": total_fees,
        "fees_pct": total_fees / gross_pnl * 100 if gross_pnl > 0 else 0,
        "wr": wr,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "pf": pf,
        "max_dd": max_dd,
        "avg_pnl": total_pnl / len(trades),
    }

def print_separator(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

def main():

    instances = {}

    main_data = {
        "name": "main_old",
        "trades": [],
        "positions": [],
        "cash": 0,
        "initial": 1000,
        "mtm_equity": None,
        "n_trades": 0,
    }
    cp_path = os.path.join(DATA_DIR, "checkpoint.json")
    if os.path.exists(cp_path):
        with open(cp_path) as f:
            cp = json.load(f)
        main_data["trades"] = cp.get("trades", [])
        main_data["positions"] = cp.get("positions", [])
        main_data["cash"] = cp.get("cash", 0)
        main_data["n_trades"] = len(main_data["trades"])
        for t in main_data["trades"]:
            t["cat_fixed"] = classify_category(t.get("market_question", ""))
    instances["main_old"] = main_data

    for dirname in os.listdir(DATA_DIR):
        full = os.path.join(DATA_DIR, dirname)
        if os.path.isdir(full) and dirname not in (".", ".."):
            inst = load_instance(full, dirname)
            if inst and inst["n_trades"] > 0:
                instances[dirname] = inst

    main_raw_path = os.path.join(DATA_DIR, "main_raw")
    if os.path.isdir(main_raw_path):
        inst = load_instance(main_raw_path, "main_raw")
        if inst:
            instances["main_raw"] = inst

    if "main_old" in instances and "main_raw" in instances:
        combined_trades = instances["main_old"]["trades"] + instances["main_raw"]["trades"]
        instances["main_combined"] = {
            "name": "main_combined",
            "trades": combined_trades,
            "positions": instances["main_raw"]["positions"],
            "cash": instances["main_raw"]["cash"],
            "initial": 1000,
            "mtm_equity": instances["main_raw"].get("mtm_equity"),
            "n_trades": len(combined_trades),
        }

    print_separator("1. OVERVIEW — ALL INSTANCES")
    print(f"\n{'Instance':<18} {'Trades':>6} {'PnL':>10} {'WR%':>6} {'AvgPnL':>8} {'MaxDD':>8} {'Fees':>8} {'PF':>6}")
    print("-" * 82)

    for name in sorted(instances.keys()):
        inst = instances[name]
        stats = analyze_trades(inst["trades"])
        if stats["n"] == 0:
            continue
        pf_str = f"{stats['pf']:.2f}" if stats["pf"] < 100 else "inf"
        print(f"{name:<18} {stats['n']:>6} {stats['pnl']:>+10.2f} {stats['wr']:>5.1f}% {stats['avg_pnl']:>+8.2f} {stats['max_dd']:>8.2f} {stats['fees']:>8.2f} {pf_str:>6}")

    main_key = "main_combined" if "main_combined" in instances else "main_old"
    main = instances[main_key]
    trades = main["trades"]

    print_separator(f"2. MAIN ({main_key}) — PnL BY CATEGORY")

    cat_trades = defaultdict(list)
    for t in trades:
        cat_trades[t["cat_fixed"]].append(t)

    print(f"\n{'Category':<15} {'Trades':>6} {'PnL':>10} {'WR%':>6} {'AvgPnL':>8} {'AvgWin':>8} {'AvgLoss':>9} {'PF':>6}")
    print("-" * 78)

    for cat in sorted(cat_trades.keys(), key=lambda c: sum(t["pnl"] for t in cat_trades[c]), reverse=True):
        stats = analyze_trades(cat_trades[cat])
        pf_str = f"{stats['pf']:.2f}" if stats["pf"] < 100 else "inf"
        print(f"{cat:<15} {stats['n']:>6} {stats['pnl']:>+10.2f} {stats['wr']:>5.1f}% {stats['avg_pnl']:>+8.2f} {stats['avg_win']:>+8.2f} {stats['avg_loss']:>+9.2f} {pf_str:>6}")

    print_separator("3. MAIN — PnL BY PRICE ZONE (entry price)")

    zones = {
        "0-10¢": (0, 0.10),
        "10-20¢": (0.10, 0.20),
        "20-30¢": (0.10, 0.30),
        "30-50¢": (0.30, 0.50),
        "50-70¢": (0.50, 0.70),
        "70-90¢": (0.70, 0.90),
        "90-100¢": (0.90, 1.00),
    }

    print(f"\n{'Zone':<12} {'Trades':>6} {'PnL':>10} {'WR%':>6} {'AvgPnL':>8} {'Fees':>8}")
    print("-" * 56)

    for zone_name, (lo, hi) in zones.items():
        zone_trades = [t for t in trades if lo <= t["entry_price"] < hi]
        if not zone_trades:
            continue
        stats = analyze_trades(zone_trades)
        print(f"{zone_name:<12} {stats['n']:>6} {stats['pnl']:>+10.2f} {stats['wr']:>5.1f}% {stats['avg_pnl']:>+8.2f} {stats['fees']:>8.2f}")

    print_separator("4. MAIN — PnL BY EXIT REASON")

    exit_groups = defaultdict(list)
    for t in trades:
        reason = t.get("exit_reason", "unknown")
        if "MR target" in reason:
            group = "MR target hit"
        elif "Trailing" in reason:
            group = "Trailing exit"
        elif "Edge gone" in reason:
            group = "Edge gone"
        elif "Resolution win" in reason:
            group = "Resolution WIN"
        elif "Resolution loss" in reason:
            group = "Resolution LOSS"
        elif "Time exit" in reason:
            group = "Time exit (12h)"
        elif "Adverse move" in reason:
            group = "Adverse move"
        elif "daily stop" in reason:
            group = "Daily stop"
        else:
            group = reason[:30]
        exit_groups[group].append(t)

    print(f"\n{'Exit Reason':<22} {'Trades':>6} {'PnL':>10} {'WR%':>6} {'AvgPnL':>8}")
    print("-" * 58)

    for group in sorted(exit_groups.keys(), key=lambda g: sum(t["pnl"] for t in exit_groups[g]), reverse=True):
        stats = analyze_trades(exit_groups[group])
        print(f"{group:<22} {stats['n']:>6} {stats['pnl']:>+10.2f} {stats['wr']:>5.1f}% {stats['avg_pnl']:>+8.2f}")

    print_separator("5. MAIN — PnL BY SIDE")

    for side in ["YES", "NO"]:
        side_trades = [t for t in trades if t["side"] == side]
        if side_trades:
            stats = analyze_trades(side_trades)
            print(f"{side}: {stats['n']} trades, PnL={stats['pnl']:+.2f}, WR={stats['wr']:.1f}%, Avg={stats['avg_pnl']:+.2f}")

    print_separator("6. TOP 10 WINNERS")

    sorted_by_pnl = sorted(trades, key=lambda t: t["pnl"], reverse=True)
    print(f"\n{'PnL':>8} {'Cat':<12} {'Side':<4} {'Entry':>6} {'Exit':>6} {'Reason':<25} {'Question':<40}")
    print("-" * 110)
    for t in sorted_by_pnl[:10]:
        print(f"{t['pnl']:>+8.2f} {t['cat_fixed']:<12} {t['side']:<4} {t['entry_price']:>6.3f} {t['exit_price']:>6.3f} {t.get('exit_reason','')[:25]:<25} {t['market_question'][:40]}")

    print_separator("7. TOP 10 LOSERS")
    print(f"\n{'PnL':>8} {'Cat':<12} {'Side':<4} {'Entry':>6} {'Exit':>6} {'Reason':<25} {'Question':<40}")
    print("-" * 110)
    for t in sorted_by_pnl[-10:]:
        print(f"{t['pnl']:>+8.2f} {t['cat_fixed']:<12} {t['side']:<4} {t['entry_price']:>6.3f} {t['exit_price']:>6.3f} {t.get('exit_reason','')[:25]:<25} {t['market_question'][:40]}")

    print_separator("8. HOLDING TIME ANALYSIS")

    hold_times = []
    for t in trades:
        try:
            entry = datetime.fromisoformat(t["entry_time"].replace("+00:00", "+00:00"))
            exit_ = datetime.fromisoformat(t["exit_time"].replace("+00:00", "+00:00"))
            hold_h = (exit_ - entry).total_seconds() / 3600
            hold_times.append((hold_h, t["pnl"], t["cat_fixed"]))
        except:
            pass

    if hold_times:
        hold_times.sort()
        holds = [h[0] for h in hold_times]
        print(f"\nMedian hold: {sorted(holds)[len(holds)//2]:.1f}h")
        print(f"Mean hold:   {sum(holds)/len(holds):.1f}h")
        print(f"Min hold:    {min(holds):.2f}h")
        print(f"Max hold:    {max(holds):.1f}h")

        buckets = {"<1h": (0, 1), "1-3h": (1, 3), "3-6h": (3, 6), "6-12h": (6, 12), "12h+": (12, 999)}
        print(f"\n{'Hold Time':<10} {'Trades':>6} {'PnL':>10} {'WR%':>6}")
        print("-" * 38)
        for bname, (lo, hi) in buckets.items():
            bucket = [(h, p) for h, p, c in hold_times if lo <= h < hi]
            if bucket:
                n = len(bucket)
                pnl = sum(p for _, p in bucket)
                wr = sum(1 for _, p in bucket if p > 0) / n * 100
                print(f"{bname:<10} {n:>6} {pnl:>+10.2f} {wr:>5.1f}%")

    print_separator("9. CATEGORY × PRICE ZONE (PnL)")

    cats_ordered = sorted(cat_trades.keys(), key=lambda c: sum(t["pnl"] for t in cat_trades[c]), reverse=True)
    simple_zones = {"lo(0-30)": (0, 0.30), "mid(30-70)": (0.30, 0.70), "hi(70-100)": (0.70, 1.0)}

    header = f"{'Category':<15}" + "".join(f"{z:>14}" for z in simple_zones.keys()) + f"{'TOTAL':>14}"
    print(f"\n{header}")
    print("-" * (15 + 14 * (len(simple_zones) + 1)))

    for cat in cats_ordered:
        row = f"{cat:<15}"
        for zname, (lo, hi) in simple_zones.items():
            zt = [t for t in cat_trades[cat] if lo <= t["entry_price"] < hi]
            pnl = sum(t["pnl"] for t in zt)
            n = len(zt)
            row += f"{pnl:>+8.1f}({n:>2})" if n > 0 else f"{'—':>14}"

        total = sum(t["pnl"] for t in cat_trades[cat])
        row += f"{total:>+8.1f}({len(cat_trades[cat]):>2})"
        print(row)

    print_separator("10. A/B TEST COMPARISON")

    ab_names = ["sports_only", "inverse", "high_edge", "quarter_kelly",
                "short_time", "mid_price", "no_politics", "meta_filter",
                "rule_based", "sports_price", "sports_ev", "price_filter"]

    print(f"\n{'Instance':<16} {'Trades':>6} {'PnL':>10} {'WR%':>6} {'AvgPnL':>8} {'MaxDD':>8} {'Best Cat':<12}")
    print("-" * 80)

    for name in ab_names:
        if name not in instances:
            continue
        inst = instances[name]
        if inst["n_trades"] == 0:
            continue
        stats = analyze_trades(inst["trades"])

        ab_cats = defaultdict(float)
        for t in inst["trades"]:
            ab_cats[t["cat_fixed"]] += t["pnl"]
        best_cat = max(ab_cats, key=ab_cats.get) if ab_cats else "—"

        print(f"{name:<16} {stats['n']:>6} {stats['pnl']:>+10.2f} {stats['wr']:>5.1f}% {stats['avg_pnl']:>+8.2f} {stats['max_dd']:>8.2f} {best_cat:<12}")

    print_separator("11. MARKETS TRADED MULTIPLE TIMES")

    market_trades = defaultdict(list)
    for t in trades:
        market_trades[t["market_question"][:60]].append(t)

    repeat_markets = {k: v for k, v in market_trades.items() if len(v) >= 3}

    if repeat_markets:
        print(f"\n{'Market':<50} {'Trades':>6} {'PnL':>10} {'WR%':>6}")
        print("-" * 76)
        for q in sorted(repeat_markets.keys(), key=lambda x: sum(t["pnl"] for t in repeat_markets[x]), reverse=True):
            ts = repeat_markets[q]
            pnl = sum(t["pnl"] for t in ts)
            wr = sum(1 for t in ts if t["pnl"] > 0) / len(ts) * 100
            print(f"{q:<50} {len(ts):>6} {pnl:>+10.2f} {wr:>5.1f}%")

    print_separator("12. SUMMARY & RECOMMENDATIONS")

    stats = analyze_trades(trades)
    print(f"\nTotal trades: {stats['n']}")
    print(f"Total PnL:    ${stats['pnl']:+.2f}")
    print(f"Win rate:     {stats['wr']:.1f}%")
    print(f"Avg PnL:      ${stats['avg_pnl']:+.2f}/trade")
    print(f"Profit factor: {stats['pf']:.2f}")
    print(f"Max drawdown:  ${stats['max_dd']:.2f}")
    print(f"Total fees:    ${stats['fees']:.2f} ({stats['fees_pct']:.0f}% of gross)")

    best_cat = max(cat_trades.keys(), key=lambda c: sum(t["pnl"] for t in cat_trades[c]))
    worst_cat = min(cat_trades.keys(), key=lambda c: sum(t["pnl"] for t in cat_trades[c]))
    print(f"\nBest category:  {best_cat} (${sum(t['pnl'] for t in cat_trades[best_cat]):+.2f})")
    print(f"Worst category: {worst_cat} (${sum(t['pnl'] for t in cat_trades[worst_cat]):+.2f})")

    print("\n--- ACTIONABLE ---")
    for cat in cats_ordered:
        pnl = sum(t["pnl"] for t in cat_trades[cat])
        n = len(cat_trades[cat])
        if pnl > 0:
            print(f"  ✓ {cat}: KEEP (${pnl:+.2f}, {n} trades)")
        elif n >= 5:
            print(f"  ✗ {cat}: EXCLUDE (${pnl:+.2f}, {n} trades)")
        else:
            print(f"  ? {cat}: INSUFFICIENT DATA (${pnl:+.2f}, {n} trades)")

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")
    main()
