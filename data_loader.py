"""Load OANDA, Alpaca, and Solana trading logs into DataFrames.

Reads from Parquet cache when available (fast), falls back to raw JSONL.
"""

import json
from pathlib import Path
import pandas as pd

CACHE_DIR = Path(__file__).parent / "cache"


# ---------------------------------------------------------------------------
# Parquet helpers
# ---------------------------------------------------------------------------

def _read_cached_nav(system: str) -> pd.DataFrame | None:
    """Return cached NAV DataFrame or None if cache miss."""
    parquet = CACHE_DIR / system / "nav.parquet"
    if parquet.exists():
        df = pd.read_parquet(parquet)
        if not df.empty:
            return df
    return None


def _read_cached_instrument_pnl(
    system: str, freq: str | None,
) -> pd.DataFrame | None:
    """Return instrument P&L from cache, applying resample/diff at read time."""
    parquet = CACHE_DIR / system / "instrument_pnl.parquet"
    if not parquet.exists():
        return None

    df = pd.read_parquet(parquet)
    if df.empty or len(df) < 2:
        return pd.DataFrame() if df.empty else None

    if freq:
        df = df.resample(freq).last().dropna(how="all")

    df = df.fillna(0.0)
    pnl = df.diff().iloc[1:]

    if freq == "D":
        pnl.index = pnl.index.date

    return pnl


def _read_cached_positions(system: str) -> pd.DataFrame | None:
    """Return cached latest positions or None if cache miss."""
    cached = CACHE_DIR / system / "latest_positions.json"
    if not cached.exists():
        return None
    with open(cached) as f:
        rows = json.load(f)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# OANDA
# ---------------------------------------------------------------------------

def load_oanda_nav(path: Path | str) -> pd.DataFrame:
    """Load OANDA NAV positions into a DataFrame.

    Returns DataFrame indexed by timestamp with columns:
    nav, balance, pl, unrealized_pl, financing, open_trade_count,
    margin_used, position_value.
    """
    cached = _read_cached_nav("oanda")
    if cached is not None:
        return cached

    # Fallback: parse JSONL directly
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


def extract_oanda_positions(path: Path | str) -> pd.DataFrame:
    """Extract positions from the latest OANDA snapshot."""
    cached = _read_cached_positions("oanda")
    if cached is not None:
        return cached

    # Fallback: parse JSONL directly
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
    """Compute per-instrument daily P&L from OANDA data.

    Convenience wrapper — calls load_oanda_instrument_pnl with freq="D".
    """
    return load_oanda_instrument_pnl(path, freq="D")


def load_oanda_instrument_pnl(
    path: Path | str, freq: str | None = "D",
) -> pd.DataFrame:
    """Compute per-instrument period P&L from OANDA data.

    Tracks cumulative (pl + unrealizedPL) per instrument at each snapshot,
    optionally resamples to *freq*, then diffs to get period deltas.

    Returns DataFrame with period index and one column per instrument.
    Values are period P&L deltas (suitable for cumsum / metrics).
    """
    cached = _read_cached_instrument_pnl("oanda", freq)
    if cached is not None:
        return cached

    # Fallback: parse JSONL directly
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()

    snapshots: list[tuple] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                ts = pd.to_datetime(rec["time_iso"])
                inst_pnl: dict[str, float] = {}
                for pos in rec.get("positions", []):
                    inst = pos["instrument"]
                    cum = float(pos.get("pl", "0")) + float(
                        pos.get("unrealizedPL", "0"))
                    inst_pnl[inst] = cum
                if inst_pnl:
                    snapshots.append((ts, inst_pnl))
            except (KeyError, json.JSONDecodeError, ValueError):
                continue

    if len(snapshots) < 2:
        return pd.DataFrame()

    all_instruments = sorted({i for _, sp in snapshots for i in sp})
    rows = []
    for ts, sp in snapshots:
        row = {inst: sp.get(inst, 0.0) for inst in all_instruments}
        row["timestamp"] = ts
        rows.append(row)

    df = pd.DataFrame(rows).set_index("timestamp")
    df.index = pd.to_datetime(df.index, utc=True)

    if freq:
        df = df.resample(freq).last().dropna(how="all")

    df = df.fillna(0.0)
    pnl = df.diff().iloc[1:]

    if freq == "D":
        pnl.index = pnl.index.date

    return pnl


# ---------------------------------------------------------------------------
# Alpaca
# ---------------------------------------------------------------------------

def load_alpaca_equity(path: Path | str) -> pd.DataFrame:
    """Load Alpaca equity positions into a DataFrame.

    Returns DataFrame indexed by timestamp with columns:
    equity, cash, long_market_value, short_market_value, positions_count.
    """
    cached = _read_cached_nav("alpaca")
    if cached is not None:
        return cached

    # Fallback: parse JSONL directly
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


def extract_alpaca_positions(path: Path | str) -> pd.DataFrame:
    """Extract positions from the latest Alpaca snapshot."""
    cached = _read_cached_positions("alpaca")
    if cached is not None:
        return cached

    # Fallback: parse JSONL directly
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


def load_alpaca_instrument_daily_pnl(path: Path | str) -> pd.DataFrame:
    """Compute per-instrument daily P&L from Alpaca data.

    Convenience wrapper — calls load_alpaca_instrument_pnl with freq="D".
    """
    return load_alpaca_instrument_pnl(path, freq="D")


def load_alpaca_instrument_pnl(
    path: Path | str, freq: str | None = "D",
) -> pd.DataFrame:
    """Compute per-instrument period P&L from Alpaca data.

    Tracks cumulative unrealized_pl per symbol at each snapshot,
    optionally resamples to *freq*, then diffs to get period deltas.

    Returns DataFrame with period index and one column per symbol.
    Values are period P&L deltas (suitable for cumsum / metrics).
    """
    cached = _read_cached_instrument_pnl("alpaca", freq)
    if cached is not None:
        return cached

    # Fallback: parse JSONL directly
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()

    snapshots: list[tuple] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                ts = pd.to_datetime(rec["time_iso"])
                pos_pnl: dict[str, float] = {}
                for pos in rec.get("positions", []):
                    sym = pos["symbol"]
                    # Use cumulative unrealized_pl (not intraday) for proper diffing
                    pos_pnl[sym] = float(pos.get("unrealized_pl", 0))
                if pos_pnl:
                    snapshots.append((ts, pos_pnl))
            except (KeyError, json.JSONDecodeError, ValueError):
                continue

    if len(snapshots) < 2:
        return pd.DataFrame()

    all_symbols = sorted({s for _, sp in snapshots for s in sp})
    rows = []
    for ts, sp in snapshots:
        row = {sym: sp.get(sym, 0.0) for sym in all_symbols}
        row["timestamp"] = ts
        rows.append(row)

    df = pd.DataFrame(rows).set_index("timestamp")
    df.index = pd.to_datetime(df.index, utc=True)

    if freq:
        df = df.resample(freq).last().dropna(how="all")

    df = df.fillna(0.0)
    pnl = df.diff().iloc[1:]

    if freq == "D":
        pnl.index = pnl.index.date

    return pnl


# ---------------------------------------------------------------------------
# Solana
# ---------------------------------------------------------------------------

def load_solana_nav(path: Path | str) -> pd.DataFrame:
    """Load Solana account snapshot into a DataFrame.

    Returns DataFrame indexed by timestamp with columns:
    nav, sol_balance_usd, positions_value_usd, positions_count,
    total_unrealized_pnl.
    """
    cached = _read_cached_nav("solana")
    if cached is not None:
        return cached

    # Fallback: parse JSONL directly
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
                if rec.get("event") != "solana_account_snapshot":
                    continue
                acct = rec["account"]
                rows.append({
                    "timestamp": pd.to_datetime(rec["time_iso"]),
                    "nav": float(acct["nav"]),
                    "sol_balance_usd": float(acct["sol_balance_usd"]),
                    "positions_value_usd": float(acct["positions_value_usd"]),
                    "positions_count": int(acct["positions_count"]),
                    "total_unrealized_pnl": float(
                        acct.get("total_unrealized_pnl", 0)),
                })
            except (KeyError, json.JSONDecodeError, ValueError):
                continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("timestamp")
    df.index = df.index.tz_convert("UTC")
    return df


def extract_solana_positions(path: Path | str) -> pd.DataFrame:
    """Extract positions from the latest Solana snapshot."""
    cached = _read_cached_positions("solana")
    if cached is not None:
        return cached

    # Fallback: parse JSONL directly
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
        entry = pos.get("entry_usd")
        pnl = pos.get("unrealized_pnl")
        rows.append({
            "symbol": pos.get("symbol", pos.get("mint", "")[:8]),
            "mint": pos.get("mint", ""),
            "value_usd": float(pos.get("value_usd", 0)),
            "entry_usd": float(entry) if entry is not None else 0.0,
            "unrealized_pnl": float(pnl) if pnl is not None else 0.0,
            "pnl_pct": (float(pnl) / float(entry))
                       if pnl is not None and entry else 0.0,
            "price_impact_pct": float(pos.get("price_impact_pct", 0)),
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def load_solana_instrument_pnl(
    path: Path | str, freq: str | None = "D",
) -> pd.DataFrame:
    """Compute per-instrument period P&L from Solana data.

    Tracks unrealized_pnl per symbol at each snapshot, optionally resamples
    to *freq*, then diffs to get period deltas.

    Returns DataFrame with period index and one column per symbol.
    Values are period P&L deltas (suitable for cumsum / metrics).
    """
    cached = _read_cached_instrument_pnl("solana", freq)
    if cached is not None:
        return cached

    # Fallback: parse JSONL directly
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()

    snapshots: list[tuple] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("event") != "solana_account_snapshot":
                    continue
                ts = pd.to_datetime(rec["time_iso"])
                sym_pnl: dict[str, float] = {}
                for pos in rec.get("positions", []):
                    sym = pos.get("symbol", pos.get("mint", "")[:8])
                    pnl = pos.get("unrealized_pnl")
                    sym_pnl[sym] = float(pnl) if pnl is not None else 0.0
                snapshots.append((ts, sym_pnl))
            except (KeyError, json.JSONDecodeError, ValueError):
                continue

    if len(snapshots) < 2:
        return pd.DataFrame()

    all_symbols = sorted({s for _, sp in snapshots for s in sp})
    rows = []
    for ts, sp in snapshots:
        row = {sym: sp.get(sym, 0.0) for sym in all_symbols}
        row["timestamp"] = ts
        rows.append(row)

    df = pd.DataFrame(rows).set_index("timestamp")
    df.index = pd.to_datetime(df.index, utc=True)

    if freq:
        df = df.resample(freq).last().dropna(how="all")

    df = df.fillna(0.0)
    pnl = df.diff().iloc[1:]

    if freq == "D":
        pnl.index = pnl.index.date

    return pnl


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def resample_daily(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Resample intraday data to daily, keeping last value per day."""
    if df.empty:
        return df
    return df.resample("D").last().dropna(subset=[value_col])
