"""Push pre-computed dashboard data to the Next.js public directory.

Reads from Parquet cache, computes all metrics and chart-ready
data, then writes JSON files to web/public/data/ and pushes
to git so Vercel auto-deploys with fresh data.

Usage:
    python push_to_vercel.py              # push all systems
    python push_to_vercel.py --system oanda  # push one system
    python push_to_vercel.py --dry-run    # compute but don't write
"""

import argparse
import json
import logging
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from data_loader import (
    _read_cached_nav,
    _read_cached_instrument_pnl,
    _read_cached_positions,
    resample_daily,
)
from metrics import (
    compute_all_metrics,
    compute_daily_returns,
    compute_instrument_metrics,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "web" / "public" / "data"

PHASE2_START = pd.Timestamp("2026-01-19", tz="UTC")

SYSTEM_CONFIG = {
    "oanda": {
        "value_col": "nav",
        "annual_daily": 252,
        "annual_5min": 252 * 78,
        "rolling_daily": 10,
        "rolling_5min": 288,
        "filter_start": PHASE2_START,
        "exposure_cols": {
            "Long Exposure": "long_market_value",
            "Short Exposure": "short_market_value",
            "Margin Used": "margin_used",
        },
        "start_nav_from_data": True,
    },
    "alpaca": {
        "value_col": "equity",
        "annual_daily": 252,
        "annual_5min": 252 * 78,
        "rolling_daily": 10,
        "rolling_5min": 288,
        "filter_start": None,
        "exposure_cols": {
            "Long Exposure": "long_market_value",
            "Short Exposure": "short_market_value",
        },
        "start_nav_from_data": False,
    },
    "solana": {
        "value_col": "nav",
        "annual_daily": 365,
        "annual_5min": 365 * 288,
        "rolling_daily": 10,
        "rolling_5min": 288,
        "filter_start": None,
        "exposure_cols": {
            "Token Positions": "positions_value_usd",
            "SOL Balance": "sol_balance_usd",
        },
        "start_nav_from_data": True,
    },
}


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            v = float(obj)
            if np.isnan(v) or np.isinf(v):
                return 0.0
            return v
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _ts_to_strings(index) -> list[str]:
    """Convert pandas index to ISO 8601 strings."""
    result = []
    for val in index:
        if hasattr(val, "isoformat"):
            result.append(val.isoformat())
        else:
            result.append(str(val))
    return result


def prepare_overview(
    nav_df: pd.DataFrame,
    value_col: str,
    annual_factor: int,
    system_info: dict | None = None,
) -> dict:
    """Prepare overview tab data."""
    metrics = compute_all_metrics(nav_df, value_col, annual_factor)
    returns = compute_daily_returns(nav_df, value_col)

    return {
        "metrics": metrics,
        "equity_curve": {
            "timestamps": _ts_to_strings(nav_df.index),
            "values": nav_df[value_col].tolist(),
        },
        "daily_returns": {
            "timestamps": _ts_to_strings(returns.index),
            "values": returns.tolist(),
        },
        "system_info": system_info or {},
    }


def prepare_positions(
    positions_df: pd.DataFrame,
    inst_pnl_daily: pd.DataFrame,
    inst_pnl_5min: pd.DataFrame,
    annual_daily: int,
    annual_5min: int,
    system: str,
    algo_pnl: float | None = None,
    positions_value: float | None = None,
    total_unrealized_pnl: float | None = None,
) -> dict:
    """Prepare positions tab data."""
    positions_list = positions_df.to_dict(orient="records") if not positions_df.empty else []

    metrics_daily = (
        compute_instrument_metrics(inst_pnl_daily, annual_daily)
        if not inst_pnl_daily.empty and len(inst_pnl_daily) >= 2
        else pd.DataFrame()
    )
    metrics_5min = (
        compute_instrument_metrics(inst_pnl_5min, annual_5min)
        if not inst_pnl_5min.empty and len(inst_pnl_5min) >= 2
        else pd.DataFrame()
    )

    top_bottom = []
    if not metrics_daily.empty:
        top_syms = metrics_daily.head(10)["instrument"].tolist()
        bottom_syms = metrics_daily.tail(10)["instrument"].tolist()
        top_bottom = list(dict.fromkeys(top_syms + bottom_syms))

    def _cum_pnl_series(pnl_df: pd.DataFrame, symbols: list[str]) -> dict:
        if pnl_df.empty:
            return {"timestamps": [], "series": {}}
        available = [s for s in symbols if s in pnl_df.columns]
        if not available:
            available = list(pnl_df.columns)
        cum = pnl_df[available].cumsum()
        return {
            "timestamps": _ts_to_strings(cum.index),
            "series": {col: cum[col].tolist() for col in available},
        }

    cum_symbols = top_bottom if system == "alpaca" else list(inst_pnl_daily.columns)

    long_count = 0
    short_count = 0
    if not positions_df.empty:
        if "side" in positions_df.columns:
            long_count = len(positions_df[positions_df["side"] == "long"])
            short_count = len(positions_df[positions_df["side"] == "short"])

    return {
        "positions": positions_list,
        "instrument_metrics": {
            "daily": metrics_daily.to_dict(orient="records") if not metrics_daily.empty else [],
            "five_min": metrics_5min.to_dict(orient="records") if not metrics_5min.empty else [],
        },
        "cumulative_pnl": {
            "daily": _cum_pnl_series(inst_pnl_daily, cum_symbols),
            "five_min": _cum_pnl_series(inst_pnl_5min, cum_symbols),
        },
        "top_bottom_symbols": top_bottom,
        "summary": {
            "total": len(positions_df),
            "long": long_count,
            "short": short_count,
            "algo_pnl": algo_pnl,
            "positions_value": positions_value,
            "total_unrealized_pnl": total_unrealized_pnl,
        },
    }


def prepare_risk(
    nav_df: pd.DataFrame,
    value_col: str,
    annual_factor: int,
    rolling_window: int,
    exposure_cols: dict[str, str],
) -> dict:
    """Prepare risk tab data."""
    returns = compute_daily_returns(nav_df, value_col)

    values = nav_df[value_col]
    peak = values.expanding().max()
    dd = (values - peak) / peak

    window = min(rolling_window, len(returns) - 1)
    if window >= 3 and len(returns) >= window:
        rolling_mean = returns.rolling(window).mean()
        rolling_std = returns.rolling(window).std()
        rolling = (rolling_mean / rolling_std * (252 ** 0.5)).dropna()
        rolling_sharpe = {
            "timestamps": _ts_to_strings(rolling.index),
            "values": [0.0 if (np.isnan(v) or np.isinf(v)) else float(v)
                       for v in rolling],
        }
    else:
        rolling_sharpe = {"timestamps": [], "values": []}

    exposure = {"timestamps": _ts_to_strings(nav_df.index), "series": {}}
    for label, col in exposure_cols.items():
        if col in nav_df.columns:
            vals = nav_df[col].abs().tolist()
            exposure["series"][label] = vals

    avg_ret = float(returns.mean()) if len(returns) > 0 else 0.0
    vol = float(returns.std()) if len(returns) > 0 else 0.0

    return {
        "drawdown": {
            "timestamps": _ts_to_strings(dd.index),
            "values": dd.tolist(),
        },
        "rolling_sharpe": rolling_sharpe,
        "exposure": exposure,
        "risk_metrics": {
            "calmar": float(compute_all_metrics(nav_df, value_col, annual_factor)["calmar"]),
            "annualized_return": float(returns.mean() * annual_factor) if len(returns) > 0 else 0.0,
            "avg_return": avg_ret,
            "volatility": vol,
        },
    }


def _write_json(path: str, data: dict) -> None:
    """Write JSON file to web/public/data/."""
    out_path = DATA_DIR / path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    body = json.dumps(data, cls=NumpyEncoder)
    out_path.write_text(body)
    logger.info("Wrote %s (%d bytes)", out_path.relative_to(REPO_ROOT), len(body))


def push_system(system: str, dry_run: bool = False) -> None:
    """Compute and push all data for one trading system."""
    cfg = SYSTEM_CONFIG[system]
    value_col = cfg["value_col"]

    raw_df = _read_cached_nav(system)
    if raw_df is None or raw_df.empty:
        logger.warning("No NAV data for %s, skipping", system)
        return

    positions_df = _read_cached_positions(system)
    if positions_df is None:
        positions_df = pd.DataFrame()

    inst_pnl_daily = _read_cached_instrument_pnl(system, freq="D")
    if inst_pnl_daily is None:
        inst_pnl_daily = pd.DataFrame()

    inst_pnl_5min = _read_cached_instrument_pnl(system, freq=None)
    if inst_pnl_5min is None:
        inst_pnl_5min = pd.DataFrame()

    if cfg["filter_start"] is not None:
        raw_df = raw_df[raw_df.index >= cfg["filter_start"]]
        if not inst_pnl_daily.empty:
            inst_pnl_daily = inst_pnl_daily[inst_pnl_daily.index >= cfg["filter_start"].date()]
        if not inst_pnl_5min.empty:
            inst_pnl_5min = inst_pnl_5min[inst_pnl_5min.index >= cfg["filter_start"]]

    if raw_df.empty or len(raw_df) < 2:
        logger.warning("Insufficient data for %s after filtering", system)
        return

    daily_df = resample_daily(raw_df, value_col)

    start_nav = float(raw_df[value_col].iloc[0]) if cfg["start_nav_from_data"] else None
    system_info = {"name": system, "start_nav": start_nav}

    algo_pnl = None
    if cfg["start_nav_from_data"] and start_nav is not None:
        algo_pnl = float(raw_df[value_col].iloc[-1]) - start_nav

    positions_value = None
    total_unrealized_pnl = None
    if system == "solana" and not positions_df.empty:
        if "value_usd" in positions_df.columns:
            positions_value = float(positions_df["value_usd"].sum())
        if "unrealized_pnl" in positions_df.columns:
            total_unrealized_pnl = float(positions_df["unrealized_pnl"].sum())

    overview_daily = prepare_overview(daily_df, value_col, cfg["annual_daily"], system_info)
    overview_5min = prepare_overview(raw_df, value_col, cfg["annual_5min"], system_info)

    positions_data = prepare_positions(
        positions_df, inst_pnl_daily, inst_pnl_5min,
        cfg["annual_daily"], cfg["annual_5min"], system,
        algo_pnl=algo_pnl,
        positions_value=positions_value,
        total_unrealized_pnl=total_unrealized_pnl,
    )

    risk_daily = prepare_risk(
        daily_df, value_col, cfg["annual_daily"],
        cfg["rolling_daily"], cfg["exposure_cols"],
    )
    risk_5min = prepare_risk(
        raw_df, value_col, cfg["annual_5min"],
        cfg["rolling_5min"], cfg["exposure_cols"],
    )

    files = {
        f"{system}/overview-daily.json": overview_daily,
        f"{system}/overview-5min.json": overview_5min,
        f"{system}/positions.json": positions_data,
        f"{system}/risk-daily.json": risk_daily,
        f"{system}/risk-5min.json": risk_5min,
    }

    for path, data in files.items():
        if dry_run:
            size = len(json.dumps(data, cls=NumpyEncoder))
            logger.info("[DRY RUN] %s: %d bytes", path, size)
        else:
            _write_json(path, data)


def _git_push() -> None:
    """Commit and push data files to trigger Vercel auto-deploy."""
    data_rel = DATA_DIR.relative_to(REPO_ROOT)
    try:
        # Check if there are changes to commit
        result = subprocess.run(
            ["git", "status", "--porcelain", str(data_rel)],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        if not result.stdout.strip():
            logger.info("No data changes to push")
            return

        subprocess.run(
            ["git", "add", str(data_rel)],
            cwd=REPO_ROOT, check=True, capture_output=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "data: update dashboard JSON"],
            cwd=REPO_ROOT, check=True, capture_output=True,
        )
        subprocess.run(
            ["git", "push"],
            cwd=REPO_ROOT, check=True, capture_output=True, timeout=30,
        )
        logger.info("Pushed data update to git (triggers Vercel deploy)")
    except subprocess.TimeoutExpired:
        logger.warning("Git push timed out")
    except subprocess.CalledProcessError as e:
        logger.warning("Git push failed: %s", e.stderr or e)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Push dashboard data")
    parser.add_argument("--system", choices=["oanda", "alpaca", "solana"],
                        help="Push only this system")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute data but don't write")
    args = parser.parse_args()

    systems = [args.system] if args.system else ["oanda", "alpaca", "solana"]
    for system in systems:
        try:
            push_system(system, dry_run=args.dry_run)
        except Exception:
            logger.exception("Failed to push %s", system)

    if not args.dry_run:
        _git_push()


if __name__ == "__main__":
    main()
