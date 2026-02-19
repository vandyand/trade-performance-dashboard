"""Parquet cache layer for JSONL trading logs.

Reads each JSONL file in a single pass, extracts all data products
(nav, instrument P&L, latest positions), and writes compact Parquet
files.  Supports incremental updates via byte-offset tracking so only
new records are parsed on each refresh.
"""

import json
import logging
import os
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent / "cache"

SYSTEMS = {
    "oanda": {
        "jsonl_path": Path(
            "/home/kingjames/rl-trader/forex-rl/logs/oanda_nav_positions.jsonl"
        ),
    },
    "alpaca": {
        "jsonl_path": Path(
            "/home/kingjames/rl-trader/forex-rl/logs/alpaca_equity_positions.jsonl"
        ),
    },
    "solana": {
        "jsonl_path": Path(
            "/home/kingjames/rl-trader/altmemecoins/logs/solana_account_positions.jsonl"
        ),
    },
    "kalshi": {
        "jsonl_path": Path(
            "/home/kingjames/kalshi-edge/logs/kalshi_nav_snapshots.jsonl"
        ),
    },
}

# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _read_jsonl_from_offset(
    path: Path, byte_offset: int = 0,
) -> tuple[list[dict], int]:
    """Read JSONL starting from *byte_offset*.

    Returns (parsed_records, new_byte_offset).
    """
    if not path.exists():
        return [], 0

    file_size = path.stat().st_size
    if byte_offset > file_size:
        # File was truncated/rotated — need full rebuild
        byte_offset = 0

    records: list[dict] = []
    with open(path, "rb") as f:
        f.seek(byte_offset)
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except (json.JSONDecodeError, ValueError):
                continue
        new_offset = f.tell()
    return records, new_offset


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, engine="pyarrow")
    os.replace(tmp, path)


def _atomic_write_json(data, path: Path) -> None:
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, path)


def _load_meta(system_dir: Path) -> dict:
    meta_path = system_dir / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    return {"byte_offset": 0}


def _save_meta(system_dir: Path, meta: dict) -> None:
    _atomic_write_json(meta, system_dir / "meta.json")


# ---------------------------------------------------------------------------
# Single-pass extractors  (one per trading system)
# ---------------------------------------------------------------------------


def _extract_oanda_data(
    records: list[dict],
) -> tuple[list[dict], list[tuple], list[dict]]:
    """Extract nav rows, instrument snapshots, and latest positions from OANDA records."""
    nav_rows: list[dict] = []
    instrument_snapshots: list[tuple] = []
    latest_positions: list[dict] = []

    for rec in records:
        try:
            acct = rec["account"]
            ts = pd.to_datetime(rec["time_iso"])

            nav_rows.append({
                "timestamp": ts,
                "nav": float(acct["NAV"]),
                "balance": float(acct["balance"]),
                "pl": float(acct["pl"]),
                "unrealized_pl": float(acct["unrealizedPL"]),
                "financing": float(acct["financing"]),
                "open_trade_count": int(acct["openTradeCount"]),
                "margin_used": float(acct["marginUsed"]),
                "position_value": float(acct["positionValue"]),
            })

            inst_pnl: dict[str, float] = {}
            pos_rows: list[dict] = []
            for pos in rec.get("positions", []):
                inst = pos["instrument"]
                cum = float(pos.get("pl", "0")) + float(
                    pos.get("unrealizedPL", "0"))
                inst_pnl[inst] = cum

                long_units = int(pos.get("long", {}).get("units", "0"))
                short_units = int(pos.get("short", {}).get("units", "0"))
                pos_rows.append({
                    "instrument": inst,
                    "net_units": long_units + short_units,
                    "long_units": long_units,
                    "short_units": short_units,
                    "pl": float(pos.get("pl", "0")),
                    "unrealized_pl": float(pos.get("unrealizedPL", "0")),
                })

            if inst_pnl:
                instrument_snapshots.append((ts, inst_pnl))
            latest_positions = pos_rows  # keep overwriting; final = latest

        except (KeyError, json.JSONDecodeError, ValueError):
            continue

    return nav_rows, instrument_snapshots, latest_positions


def _extract_alpaca_data(
    records: list[dict],
) -> tuple[list[dict], list[tuple], list[dict]]:
    """Extract equity rows, instrument snapshots, and latest positions from Alpaca records."""
    nav_rows: list[dict] = []
    instrument_snapshots: list[tuple] = []
    latest_positions: list[dict] = []

    for rec in records:
        try:
            acct = rec["account"]
            totals = rec.get("totals", {})
            ts = pd.to_datetime(rec["time_iso"])

            nav_rows.append({
                "timestamp": ts,
                "equity": float(acct["equity"]),
                "cash": float(acct["cash"]),
                "long_market_value": float(acct["long_market_value"]),
                "short_market_value": float(acct["short_market_value"]),
                "positions_count": int(totals.get("positions_count", 0)),
            })

            pos_pnl: dict[str, float] = {}
            pos_rows: list[dict] = []
            for pos in rec.get("positions", []):
                sym = pos["symbol"]
                pos_pnl[sym] = float(pos.get("unrealized_pl", 0))

                side = pos.get("side", "")
                qty = float(pos.get("qty", 0))
                if "SHORT" in side.upper():
                    qty = -qty
                pos_rows.append({
                    "symbol": sym,
                    "qty": qty,
                    "side": "short" if "SHORT" in side.upper() else "long",
                    "market_value": float(pos.get("market_value", 0)),
                    "cost_basis": float(pos.get("cost_basis", 0)),
                    "avg_entry_price": float(pos.get("avg_entry_price", 0)),
                    "unrealized_pl": float(pos.get("unrealized_pl", 0)),
                    "unrealized_plpc": float(pos.get("unrealized_plpc", 0)),
                })

            if pos_pnl:
                instrument_snapshots.append((ts, pos_pnl))
            latest_positions = pos_rows

        except (KeyError, json.JSONDecodeError, ValueError):
            continue

    return nav_rows, instrument_snapshots, latest_positions


def _extract_solana_data(
    records: list[dict],
) -> tuple[list[dict], list[tuple], list[dict]]:
    """Extract nav rows, instrument snapshots, and latest positions from Solana records."""
    nav_rows: list[dict] = []
    instrument_snapshots: list[tuple] = []
    latest_positions: list[dict] = []

    for rec in records:
        try:
            if rec.get("event") != "solana_account_snapshot":
                continue

            acct = rec["account"]
            ts = pd.to_datetime(rec["time_iso"])

            nav_rows.append({
                "timestamp": ts,
                "nav": float(acct["nav"]),
                "sol_balance_usd": float(acct["sol_balance_usd"]),
                "positions_value_usd": float(acct["positions_value_usd"]),
                "positions_count": int(acct["positions_count"]),
                "total_unrealized_pnl": float(
                    acct.get("total_unrealized_pnl", 0)),
            })

            sym_pnl: dict[str, float] = {}
            pos_rows: list[dict] = []
            for pos in rec.get("positions", []):
                sym = pos.get("symbol", pos.get("mint", "")[:8])
                pnl = pos.get("unrealized_pnl")
                sym_pnl[sym] = float(pnl) if pnl is not None else 0.0

                entry = pos.get("entry_usd")
                pos_rows.append({
                    "symbol": sym,
                    "mint": pos.get("mint", ""),
                    "value_usd": float(pos.get("value_usd", 0)),
                    "entry_usd": float(entry) if entry is not None else 0.0,
                    "unrealized_pnl": float(pnl) if pnl is not None else 0.0,
                    "pnl_pct": (float(pnl) / float(entry))
                               if pnl is not None and entry else 0.0,
                    "price_impact_pct": float(
                        pos.get("price_impact_pct", 0)),
                })

            if sym_pnl:
                instrument_snapshots.append((ts, sym_pnl))
            latest_positions = pos_rows

        except (KeyError, json.JSONDecodeError, ValueError):
            continue

    return nav_rows, instrument_snapshots, latest_positions


def _extract_kalshi_data(
    records: list[dict],
) -> tuple[list[dict], list[tuple], list[dict]]:
    """Extract nav rows, instrument snapshots, and latest positions from Kalshi records."""
    nav_rows: list[dict] = []
    instrument_snapshots: list[tuple] = []
    latest_positions: list[dict] = []

    for rec in records:
        try:
            if rec.get("event") != "kalshi_nav_snapshot":
                continue

            portfolio = rec["portfolio"]
            agents = rec.get("agents", [])
            ts = pd.to_datetime(rec["time_iso"])

            total_cash = sum(a.get("cash_cents", 0) for a in agents) / 100.0
            total_mtm = sum(a.get("mtm_value_cents", 0) for a in agents) / 100.0

            nav_rows.append({
                "timestamp": ts,
                "nav": portfolio["total_nav_cents"] / 100.0,
                "cash": total_cash,
                "positions_value": total_mtm,
                "positions_count": int(portfolio.get("total_positions", 0)),
                "settlements_count": int(portfolio.get("total_settlements", 0)),
            })

            agent_pnl: dict[str, float] = {}
            pos_rows: list[dict] = []
            for agent in agents:
                name = agent["name"]
                pnl = agent.get("pnl_cents", 0) / 100.0
                agent_pnl[name] = pnl

                pos_rows.append({
                    "symbol": name,
                    "nav_usd": agent.get("nav_cents", 0) / 100.0,
                    "cash_usd": agent.get("cash_cents", 0) / 100.0,
                    "positions_count": int(agent.get("positions_count", 0)),
                    "cost_basis_usd": agent.get("cost_basis_cents", 0) / 100.0,
                    "mtm_value_usd": agent.get("mtm_value_cents", 0) / 100.0,
                    "unrealized_pnl": pnl,
                    "pnl_pct": float(agent.get("pnl_pct", 0)),
                    "settlements": int(agent.get("settlements", 0)),
                })

            if agent_pnl:
                instrument_snapshots.append((ts, agent_pnl))
            latest_positions = pos_rows

        except (KeyError, json.JSONDecodeError, ValueError):
            continue

    return nav_rows, instrument_snapshots, latest_positions


# ---------------------------------------------------------------------------
# DataFrame builders
# ---------------------------------------------------------------------------


def _nav_rows_to_df(rows: list[dict]) -> pd.DataFrame:
    """Convert nav row dicts to a timestamp-indexed DataFrame."""
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).set_index("timestamp")
    df.index = df.index.tz_convert("UTC")
    return df


def _snapshots_to_cumulative_df(
    snapshots: list[tuple],
) -> pd.DataFrame:
    """Build wide DataFrame of cumulative per-instrument values.

    Index: UTC timestamp.  Columns: one per instrument.
    Values are cumulative (not diffed) — caller applies resample/diff.
    """
    if not snapshots:
        return pd.DataFrame()

    all_instruments = sorted({i for _, sp in snapshots for i in sp})
    rows = []
    for ts, sp in snapshots:
        row = {inst: sp.get(inst, 0.0) for inst in all_instruments}
        row["timestamp"] = ts
        rows.append(row)

    df = pd.DataFrame(rows).set_index("timestamp")
    df.index = pd.to_datetime(df.index, utc=True)
    return df


# ---------------------------------------------------------------------------
# Cache refresh
# ---------------------------------------------------------------------------

_EXTRACTORS = {
    "oanda": _extract_oanda_data,
    "alpaca": _extract_alpaca_data,
    "solana": _extract_solana_data,
    "kalshi": _extract_kalshi_data,
}


def refresh_cache(
    system: str,
    *,
    full_rebuild: bool = False,
    cache_dir: Path | None = None,
    jsonl_path: Path | None = None,
) -> None:
    """Refresh the Parquet cache for one trading system.

    Parameters
    ----------
    system : str
        One of "oanda", "alpaca", "solana".
    full_rebuild : bool
        If True, ignore saved byte offset and re-read the entire file.
    cache_dir : Path, optional
        Override the cache directory (for testing).
    jsonl_path : Path, optional
        Override the JSONL source path (for testing).
    """
    cfg = SYSTEMS[system]
    src = jsonl_path or cfg["jsonl_path"]
    dest = (cache_dir or CACHE_DIR) / system
    dest.mkdir(parents=True, exist_ok=True)

    meta = _load_meta(dest)
    offset = 0 if full_rebuild else meta.get("byte_offset", 0)

    records, new_offset = _read_jsonl_from_offset(src, offset)

    if not records:
        if offset == 0:
            # Source file is empty or missing — write empty artifacts
            _atomic_write_parquet(pd.DataFrame(), dest / "nav.parquet")
            _atomic_write_parquet(
                pd.DataFrame(), dest / "instrument_pnl.parquet")
            _atomic_write_json([], dest / "latest_positions.json")
        _save_meta(dest, {"byte_offset": new_offset})
        return

    extractor = _EXTRACTORS[system]
    nav_rows, inst_snapshots, latest_pos = extractor(records)

    new_nav_df = _nav_rows_to_df(nav_rows)
    new_pnl_df = _snapshots_to_cumulative_df(inst_snapshots)

    nav_path = dest / "nav.parquet"
    pnl_path = dest / "instrument_pnl.parquet"

    is_incremental = offset > 0

    # --- NAV ---
    if is_incremental and nav_path.exists():
        existing = pd.read_parquet(nav_path)
        combined = pd.concat([existing, new_nav_df])
    else:
        combined = new_nav_df
    if not combined.empty:
        _atomic_write_parquet(combined, nav_path)

    # --- Instrument P&L (cumulative) ---
    if is_incremental and pnl_path.exists():
        existing = pd.read_parquet(pnl_path)
        combined = pd.concat([existing, new_pnl_df]).fillna(0.0)
    else:
        combined = new_pnl_df
    if not combined.empty:
        _atomic_write_parquet(combined, pnl_path)

    # --- Latest positions (always full replace) ---
    _atomic_write_json(latest_pos, dest / "latest_positions.json")

    # --- Meta ---
    _save_meta(dest, {"byte_offset": new_offset})
    logger.info(
        "%s: cached %d new records (offset %d -> %d)",
        system, len(records), offset, new_offset,
    )


def refresh_all(*, full_rebuild: bool = False) -> None:
    """Refresh caches for all trading systems."""
    for system in SYSTEMS:
        try:
            refresh_cache(system, full_rebuild=full_rebuild)
        except Exception:
            logger.exception("Failed to refresh cache for %s", system)
