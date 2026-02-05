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
