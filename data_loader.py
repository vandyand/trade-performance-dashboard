"""Load OANDA and Alpaca JSONL trading logs into DataFrames."""

import json
from pathlib import Path
import pandas as pd


def load_oanda_nav(path: Path | str) -> pd.DataFrame:
    """Load OANDA NAV positions JSONL into a DataFrame.

    Returns DataFrame indexed by timestamp with columns:
    nav, balance, pl, unrealized_pl, financing, open_trade_count,
    margin_used, position_value.
    """
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()

    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                acct = rec["account"]
                rows.append({
                    "timestamp": pd.to_datetime(rec["time_iso"]),
                    "nav": float(acct["NAV"]),
                    "balance": float(acct["balance"]),
                    "pl": float(acct["pl"]),
                    "unrealized_pl": float(acct["unrealizedPL"]),
                    "financing": float(acct["financing"]),
                    "open_trade_count": int(acct["openTradeCount"]),
                    "margin_used": float(acct["marginUsed"]),
                    "position_value": float(acct["positionValue"]),
                })
            except (KeyError, json.JSONDecodeError, ValueError):
                continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("timestamp")
    df.index = df.index.tz_convert("UTC")
    return df


def load_alpaca_equity(path: Path | str) -> pd.DataFrame:
    """Load Alpaca equity positions JSONL into a DataFrame.

    Returns DataFrame indexed by timestamp with columns:
    equity, cash, long_market_value, short_market_value, positions_count.
    """
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()

    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                acct = rec["account"]
                totals = rec.get("totals", {})
                rows.append({
                    "timestamp": pd.to_datetime(rec["time_iso"]),
                    "equity": float(acct["equity"]),
                    "cash": float(acct["cash"]),
                    "long_market_value": float(acct["long_market_value"]),
                    "short_market_value": float(acct["short_market_value"]),
                    "positions_count": int(totals.get("positions_count", 0)),
                })
            except (KeyError, json.JSONDecodeError, ValueError):
                continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("timestamp")
    df.index = df.index.tz_convert("UTC")
    return df


def resample_daily(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Resample intraday data to daily, keeping last value per day."""
    if df.empty:
        return df
    return df.resample("D").last().dropna(subset=[value_col])


def extract_oanda_positions(path: Path | str) -> pd.DataFrame:
    """Extract positions from the latest OANDA snapshot."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()

    last_line = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                last_line = line

    if not last_line:
        return pd.DataFrame()

    rec = json.loads(last_line)
    rows = []
    for pos in rec.get("positions", []):
        long_units = int(pos.get("long", {}).get("units", "0"))
        short_units = int(pos.get("short", {}).get("units", "0"))
        rows.append({
            "instrument": pos["instrument"],
            "net_units": long_units + short_units,
            "long_units": long_units,
            "short_units": short_units,
            "pl": float(pos.get("pl", "0")),
            "unrealized_pl": float(pos.get("unrealizedPL", "0")),
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def load_oanda_instrument_daily_pnl(path: Path | str) -> pd.DataFrame:
    """Compute per-instrument daily P&L from OANDA JSONL.

    Tracks cumulative (pl + unrealizedPL) per instrument at end of each day,
    then computes daily deltas.

    Returns DataFrame indexed by date with one column per instrument,
    values are daily P&L in dollars.
    """
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()

    daily_snapshots = {}

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                ts = pd.to_datetime(rec["time_iso"])
                date_key = ts.date()
                inst_pnl = {}
                for pos in rec.get("positions", []):
                    inst = pos["instrument"]
                    cum = float(pos.get("pl", "0")) + float(pos.get("unrealizedPL", "0"))
                    inst_pnl[inst] = cum
                if inst_pnl:
                    daily_snapshots[date_key] = inst_pnl
            except (KeyError, json.JSONDecodeError, ValueError):
                continue

    if len(daily_snapshots) < 2:
        return pd.DataFrame()

    dates = sorted(daily_snapshots.keys())
    all_instruments = sorted({i for s in daily_snapshots.values() for i in s})

    cum_rows = []
    for d in dates:
        row = {inst: daily_snapshots[d].get(inst, 0.0) for inst in all_instruments}
        row["date"] = d
        cum_rows.append(row)

    cum_df = pd.DataFrame(cum_rows).set_index("date")
    daily_pnl = cum_df.diff().iloc[1:]
    return daily_pnl


def load_alpaca_instrument_daily_pnl(path: Path | str) -> pd.DataFrame:
    """Compute per-instrument daily P&L from Alpaca JSONL.

    Uses unrealized_intraday_pl from the last snapshot of each trading day.

    Returns DataFrame indexed by date with one column per symbol,
    values are daily P&L in dollars.
    """
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()

    daily_snapshots = {}

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                ts = pd.to_datetime(rec["time_iso"])
                date_key = ts.date()
                pos_pnl = {}
                for pos in rec.get("positions", []):
                    sym = pos["symbol"]
                    pos_pnl[sym] = float(pos.get("unrealized_intraday_pl", 0))
                if pos_pnl:
                    daily_snapshots[date_key] = pos_pnl
            except (KeyError, json.JSONDecodeError, ValueError):
                continue

    if not daily_snapshots:
        return pd.DataFrame()

    dates = sorted(daily_snapshots.keys())
    all_symbols = sorted({s for snap in daily_snapshots.values() for s in snap})

    rows = []
    for d in dates:
        row = {sym: daily_snapshots[d].get(sym, 0.0) for sym in all_symbols}
        row["date"] = d
        rows.append(row)

    return pd.DataFrame(rows).set_index("date")


def extract_alpaca_positions(path: Path | str) -> pd.DataFrame:
    """Extract positions from the latest Alpaca snapshot."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()

    last_line = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                last_line = line

    if not last_line:
        return pd.DataFrame()

    rec = json.loads(last_line)
    rows = []
    for pos in rec.get("positions", []):
        side = pos.get("side", "")
        qty = float(pos.get("qty", 0))
        if "SHORT" in side.upper():
            qty = -qty
        rows.append({
            "symbol": pos["symbol"],
            "qty": qty,
            "side": "short" if "SHORT" in side.upper() else "long",
            "market_value": float(pos.get("market_value", 0)),
            "cost_basis": float(pos.get("cost_basis", 0)),
            "avg_entry_price": float(pos.get("avg_entry_price", 0)),
            "unrealized_pl": float(pos.get("unrealized_pl", 0)),
            "unrealized_plpc": float(pos.get("unrealized_plpc", 0)),
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame()
