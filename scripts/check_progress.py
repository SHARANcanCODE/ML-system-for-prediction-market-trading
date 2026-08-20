from __future__ import annotations

import os
import sys
import time
from pathlib import Path

PROGRESS_DIR = Path("data/raw/.progress")
RAW_DIR = Path("data/raw")

def get_status():

    status_file = PROGRESS_DIR / "STATUS.txt"
    if status_file.exists() and (time.time() - status_file.stat().st_mtime) < 120:
        return status_file.read_text()

    lines = []
    lines.append(f"Collection Progress — {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 60)

    totals = {
        "trades": 500,
        "prices_f60": 1000,
        "orderbooks": 1000,
    }

    for name, expected in totals.items():
        done_file = PROGRESS_DIR / f"{name}.done"
        if done_file.exists():
            done = len(done_file.read_text().strip().split("\n"))
        else:
            done = 0
        pct = done / expected * 100 if expected else 0
        bar_len = 30
        filled = int(bar_len * done / expected) if expected else 0
        bar = "#" * filled + "." * (bar_len - filled)
        lines.append(f"  {name:15s} [{bar}] {done:>5}/{expected} ({pct:.0f}%)")

    lines.append("")
    lines.append("Output files:")
    for pattern, label in [
        ("trades/trades_incremental_*.jsonl", "trades JSONL"),
        ("prices/prices_f60_incremental_*.jsonl", "prices JSONL"),
        ("orderbooks/orderbooks_full_*.jsonl", "orderbooks JSONL"),
    ]:
        files = list(RAW_DIR.glob(pattern))
        if files:
            total_size = sum(f.stat().st_size for f in files)
            mb = total_size / (1024 * 1024)
            lines.append(f"  {label:20s} {len(files)} files, {mb:.1f} MB")

    lines.append("")
    pid_check = os.popen("pgrep -f 'collect_all.py' 2>/dev/null").read().strip()
    if pid_check:
        lines.append(f"Process: RUNNING (PID {pid_check})")
    else:
        lines.append("Process: NOT RUNNING")

    return "\n".join(lines)

def main():
    watch = "--watch" in sys.argv or "-w" in sys.argv

    if watch:
        try:
            while True:
                os.system("clear")
                print(get_status())
                print(f"\n(refreshing every 10s, Ctrl+C to stop)")
                time.sleep(10)
        except KeyboardInterrupt:
            pass
    else:
        print(get_status())

if __name__ == "__main__":
    main()
