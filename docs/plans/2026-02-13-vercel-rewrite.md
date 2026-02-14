# Trading Dashboard Vercel Rewrite — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rewrite the Streamlit trading dashboard as a Next.js app on Vercel with instant navigation, Lightweight Charts, and data pushed to Vercel Blob.

**Architecture:** Python push script reads Parquet cache, pre-computes all metrics/chart data, and uploads JSON to Vercel Blob every 5 min via systemd timer. Next.js static app fetches JSON via SWR with 60s revalidation. Three URL-based routes per system, three sub-routes per tab.

**Tech Stack:** Next.js 14 (App Router), Tailwind CSS, lightweight-charts v4, SWR, Vercel Blob, Python (push script)

---

## Project Structure

```
trade-performance-dashboard/
├── web/                          ← Next.js app (deployed to Vercel)
│   ├── app/
│   │   ├── layout.tsx
│   │   ├── page.tsx
│   │   └── [system]/
│   │       ├── layout.tsx
│   │       ├── page.tsx          ← Overview tab
│   │       ├── positions/page.tsx
│   │       └── risk/page.tsx
│   ├── components/
│   │   ├── charts/
│   │   │   ├── EquityCurve.tsx
│   │   │   ├── DrawdownChart.tsx
│   │   │   ├── DailyReturns.tsx
│   │   │   ├── RollingSharpe.tsx
│   │   │   ├── CumulativePnL.tsx
│   │   │   ├── ExposureChart.tsx
│   │   │   └── ChartContainer.tsx
│   │   ├── HorizontalBar.tsx
│   │   ├── MetricCard.tsx
│   │   ├── SystemSidebar.tsx
│   │   ├── TimeframeToggle.tsx
│   │   └── DataTable.tsx
│   ├── lib/
│   │   ├── types.ts
│   │   ├── data.ts
│   │   ├── systems.ts
│   │   └── format.ts
│   ├── tailwind.config.ts
│   ├── next.config.ts
│   └── package.json
├── push_to_vercel.py             ← Python push script (runs on server)
├── tests/
│   └── test_push_to_vercel.py
└── (existing Streamlit files unchanged)
```

---

## Task 1: Initialize Next.js Project

**Files:**
- Create: `web/package.json`, `web/tsconfig.json`, `web/tailwind.config.ts`, `web/next.config.ts`, `web/app/layout.tsx`, `web/app/page.tsx`, `web/.gitignore`

**Step 1: Scaffold Next.js app**

```bash
cd /home/kingjames/trade-performance-dashboard
npx create-next-app@latest web --typescript --tailwind --eslint --app --no-src-dir --import-alias "@/*"
```

Accept defaults. This creates the full Next.js scaffold.

**Step 2: Install dependencies**

```bash
cd /home/kingjames/trade-performance-dashboard/web
npm install lightweight-charts swr
npm install -D @types/node
```

**Step 3: Configure Tailwind for dark theme**

Replace `web/tailwind.config.ts`:

```typescript
import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        bg: "#0E1117",
        card: "#1E2130",
        primary: "#4A90D9",
        accent: {
          green: "#00C853",
          red: "#FF1744",
          orange: "#FF9100",
          purple: "#AA00FF",
        },
      },
    },
  },
  plugins: [],
};
export default config;
```

**Step 4: Configure next.config.ts**

```typescript
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "export",
  images: { unoptimized: true },
};
export default nextConfig;
```

Note: `output: "export"` produces a static site. We don't need SSR since all data comes from Vercel Blob client-side.

**Step 5: Verify it builds**

```bash
cd /home/kingjames/trade-performance-dashboard/web
npm run build
```

Expected: Build succeeds.

**Step 6: Commit**

```bash
git add web/
git commit -m "feat: scaffold Next.js app with Tailwind and Lightweight Charts"
```

---

## Task 2: TypeScript Types and System Config

**Files:**
- Create: `web/lib/types.ts`, `web/lib/systems.ts`, `web/lib/format.ts`

**Step 1: Define data types**

Create `web/lib/types.ts`:

```typescript
export interface TimeSeries {
  timestamps: string[];
  values: number[];
}

export interface OverviewData {
  metrics: {
    total_return: number;
    sharpe: number;
    sortino: number;
    max_drawdown: number;
    win_rate: number;
    trading_days: number;
    annualized_return: number;
    calmar: number;
  };
  equity_curve: TimeSeries;
  daily_returns: TimeSeries;
  system_info: {
    name: string;
    description: string;
    start_date: string | null;
    start_nav: number | null;
  };
}

export interface Position {
  symbol: string;
  [key: string]: string | number;
}

export interface InstrumentMetric {
  instrument: string;
  total_pnl: number;
  avg_daily_pnl: number;
  sharpe: number;
  sortino: number;
  win_rate: number;
  trading_days: number;
}

export interface PositionsData {
  positions: Position[];
  instrument_metrics: {
    daily: InstrumentMetric[];
    five_min: InstrumentMetric[];
  };
  cumulative_pnl: {
    daily: { timestamps: string[]; series: Record<string, number[]> };
    five_min: { timestamps: string[]; series: Record<string, number[]> };
  };
  top_bottom_symbols: string[];
  summary: {
    total: number;
    long: number;
    short: number;
    algo_pnl: number | null;
    positions_value: number | null;
    total_unrealized_pnl: number | null;
  };
}

export interface RiskData {
  drawdown: TimeSeries;
  rolling_sharpe: TimeSeries;
  exposure: {
    timestamps: string[];
    series: Record<string, number[]>;
  };
  risk_metrics: {
    calmar: number;
    annualized_return: number;
    avg_return: number;
    volatility: number;
  };
}

export type SystemSlug = "oanda" | "alpaca" | "solana";
export type Timeframe = "daily" | "5min";
```

**Step 2: Define system configuration**

Create `web/lib/systems.ts`:

```typescript
import { SystemSlug } from "./types";

export interface SystemConfig {
  slug: SystemSlug;
  name: string;
  label: string;
  valueName: string;
  description: string[];
  startCaption: string | null;
}

export const SYSTEMS: Record<SystemSlug, SystemConfig> = {
  oanda: {
    slug: "oanda",
    name: "OANDA Forex",
    label: "OANDA",
    valueName: "NAV",
    description: [
      "20 forex instruments",
      "TCN + Actor-Critic RL",
      "Daily position sizing",
      "Daily decision intervals",
    ],
    startCaption: "Algorithm began trading Jan 19, 2026",
  },
  alpaca: {
    slug: "alpaca",
    name: "Alpaca Equities",
    label: "Alpaca",
    valueName: "Equity",
    description: [
      "100 long/short positions",
      "US equities universe",
      "Paper trading",
      "Daily rebalancing",
    ],
    startCaption: "Algorithm began trading Feb 2, 2026 with account balance of $100,000.00",
  },
  solana: {
    slug: "solana",
    name: "Solana Altmemecoins",
    label: "Solana",
    valueName: "NAV",
    description: [
      "Solana memecoins (long-only)",
      "TD3 reinforcement learning",
      "5-minute decision intervals",
      "Jupiter DEX execution",
    ],
    startCaption: null,
  },
};

export const SYSTEM_SLUGS: SystemSlug[] = ["oanda", "alpaca", "solana"];
```

**Step 3: Formatting utilities**

Create `web/lib/format.ts`:

```typescript
export function fmtPct(value: number, decimals = 2): string {
  return `${(value * 100).toFixed(decimals)}%`;
}

export function fmtDollar(value: number, decimals = 2): string {
  const sign = value >= 0 ? "" : "-";
  return `${sign}$${Math.abs(value).toFixed(decimals)}`;
}

export function fmtNumber(value: number, decimals = 2): string {
  return value.toFixed(decimals);
}

export function fmtSignedDollar(value: number, decimals = 2): string {
  const sign = value >= 0 ? "+" : "-";
  return `${sign}$${Math.abs(value).toFixed(decimals)}`;
}
```

**Step 4: Commit**

```bash
git add web/lib/
git commit -m "feat: add TypeScript types, system config, and formatting utils"
```

---

## Task 3: Vercel Blob Setup and Push Script

**Files:**
- Create: `push_to_vercel.py`, `tests/test_push_to_vercel.py`
- Modify: `update_cache.py` (add push call)

**Step 1: Install Vercel Blob Python dependency**

```bash
cd /home/kingjames/trade-performance-dashboard
source venv/bin/activate
pip install requests
```

(requests is already installed but confirm)

**Step 2: Write failing test for push script**

Create `tests/test_push_to_vercel.py`:

```python
"""Tests for push_to_vercel data preparation (no actual uploads)."""
import pandas as pd
import numpy as np
import pytest
from push_to_vercel import prepare_overview, prepare_positions, prepare_risk


@pytest.fixture
def sample_nav_df():
    """Create sample NAV DataFrame like OANDA."""
    idx = pd.date_range("2026-01-20", periods=10, freq="D", tz="UTC")
    return pd.DataFrame({
        "nav": np.linspace(1000, 1050, 10),
        "balance": np.linspace(1000, 1040, 10),
        "pl": np.linspace(0, 50, 10),
        "unrealized_pl": np.linspace(0, 10, 10),
        "financing": np.zeros(10),
        "open_trade_count": np.full(10, 5, dtype=int),
        "margin_used": np.linspace(100, 120, 10),
        "position_value": np.linspace(500, 600, 10),
    }, index=idx)


@pytest.fixture
def sample_inst_pnl():
    """Create sample instrument P&L DataFrame."""
    idx = pd.date_range("2026-01-20", periods=10, freq="D", tz="UTC")
    return pd.DataFrame({
        "AAPL": np.random.randn(10) * 5,
        "MSFT": np.random.randn(10) * 3,
        "GOOG": np.random.randn(10) * 4,
    }, index=idx)


@pytest.fixture
def sample_positions():
    """Create sample positions DataFrame like Alpaca."""
    return pd.DataFrame([
        {"symbol": "AAPL", "qty": 10.0, "side": "long", "market_value": 1500.0,
         "cost_basis": 1400.0, "avg_entry_price": 140.0,
         "unrealized_pl": 100.0, "unrealized_plpc": 0.071},
        {"symbol": "MSFT", "qty": -5.0, "side": "short", "market_value": -2000.0,
         "cost_basis": 2100.0, "avg_entry_price": 420.0,
         "unrealized_pl": 100.0, "unrealized_plpc": 0.048},
    ])


def test_prepare_overview(sample_nav_df):
    result = prepare_overview(sample_nav_df, "nav", annual_factor=252)
    assert "metrics" in result
    assert "equity_curve" in result
    assert "daily_returns" in result
    assert len(result["equity_curve"]["timestamps"]) == 10
    assert len(result["equity_curve"]["values"]) == 10
    assert len(result["daily_returns"]["timestamps"]) == 9  # pct_change drops first
    assert isinstance(result["metrics"]["sharpe"], float)
    assert isinstance(result["metrics"]["total_return"], float)


def test_prepare_positions(sample_positions, sample_inst_pnl):
    result = prepare_positions(
        positions_df=sample_positions,
        inst_pnl_daily=sample_inst_pnl,
        inst_pnl_5min=sample_inst_pnl,
        annual_daily=252,
        annual_5min=252 * 78,
        system="alpaca",
    )
    assert "positions" in result
    assert "instrument_metrics" in result
    assert "cumulative_pnl" in result
    assert len(result["positions"]) == 2
    assert "daily" in result["instrument_metrics"]
    assert "five_min" in result["instrument_metrics"]


def test_prepare_risk(sample_nav_df):
    result = prepare_risk(
        sample_nav_df, "nav",
        annual_factor=252,
        rolling_window=5,
        exposure_cols={"Margin Used": "margin_used"},
    )
    assert "drawdown" in result
    assert "rolling_sharpe" in result
    assert "exposure" in result
    assert "risk_metrics" in result
    assert len(result["drawdown"]["timestamps"]) == 10
```

**Step 3: Run tests to verify they fail**

```bash
cd /home/kingjames/trade-performance-dashboard
source venv/bin/activate
python -m pytest tests/test_push_to_vercel.py -v
```

Expected: FAIL (push_to_vercel module not found)

**Step 4: Implement push script**

Create `push_to_vercel.py`:

```python
"""Push pre-computed dashboard data to Vercel Blob.

Reads from Parquet cache, computes all metrics and chart-ready
data, then uploads JSON to Vercel Blob via REST API.

Usage:
    python push_to_vercel.py              # push all systems
    python push_to_vercel.py --system oanda  # push one system
    python push_to_vercel.py --dry-run    # compute but don't upload
"""

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests

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

BLOB_TOKEN = os.environ.get("BLOB_READ_WRITE_TOKEN", "")
BLOB_API = "https://blob.vercel-storage.com"

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

    # Instrument metrics for both timeframes
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

    # Top/bottom symbols (from daily metrics)
    top_bottom = []
    if not metrics_daily.empty:
        top_syms = metrics_daily.head(10)["instrument"].tolist()
        bottom_syms = metrics_daily.tail(10)["instrument"].tolist()
        top_bottom = list(dict.fromkeys(top_syms + bottom_syms))

    # Cumulative P&L series
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

    # For OANDA, use all instruments; for Alpaca, use top/bottom
    cum_symbols = top_bottom if system == "alpaca" else list(inst_pnl_daily.columns)

    # Position summary
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

    # Drawdown
    values = nav_df[value_col]
    peak = values.expanding().max()
    dd = (values - peak) / peak

    # Rolling Sharpe
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

    # Exposure
    exposure = {"timestamps": _ts_to_strings(nav_df.index), "series": {}}
    for label, col in exposure_cols.items():
        if col in nav_df.columns:
            vals = nav_df[col].abs().tolist()
            exposure["series"][label] = vals

    # Risk metrics
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


def _upload_blob(path: str, data: dict) -> str | None:
    """Upload JSON to Vercel Blob. Returns URL or None on failure."""
    if not BLOB_TOKEN:
        logger.warning("BLOB_READ_WRITE_TOKEN not set, skipping upload for %s", path)
        return None

    body = json.dumps(data, cls=NumpyEncoder)
    resp = requests.put(
        f"{BLOB_API}/{path}",
        headers={
            "Authorization": f"Bearer {BLOB_TOKEN}",
            "x-api-version": "7",
            "content-type": "application/json",
        },
        data=body,
    )

    if resp.status_code == 200:
        url = resp.json().get("url")
        logger.info("Uploaded %s (%d bytes) -> %s", path, len(body), url)
        return url
    else:
        logger.error("Upload failed for %s: %s %s", path, resp.status_code, resp.text)
        return None


def push_system(system: str, dry_run: bool = False) -> None:
    """Compute and push all data for one trading system."""
    cfg = SYSTEM_CONFIG[system]
    value_col = cfg["value_col"]

    # Load data from Parquet cache
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

    # Apply OANDA phase2 filter
    if cfg["filter_start"] is not None:
        raw_df = raw_df[raw_df.index >= cfg["filter_start"]]
        if not inst_pnl_daily.empty:
            inst_pnl_daily = inst_pnl_daily[inst_pnl_daily.index >= cfg["filter_start"].date()]
        if not inst_pnl_5min.empty:
            inst_pnl_5min = inst_pnl_5min[inst_pnl_5min.index >= cfg["filter_start"]]

    if raw_df.empty or len(raw_df) < 2:
        logger.warning("Insufficient data for %s after filtering", system)
        return

    # Prepare daily and 5-min overview variants
    daily_df = resample_daily(raw_df, value_col)

    # System info
    start_nav = float(raw_df[value_col].iloc[0]) if cfg["start_nav_from_data"] else None
    system_info = {
        "name": system,
        "start_nav": start_nav,
    }

    # Algo P&L (for OANDA)
    algo_pnl = None
    if cfg["start_nav_from_data"] and start_nav is not None:
        algo_pnl = float(raw_df[value_col].iloc[-1]) - start_nav

    # Positions value / unrealized PnL (for Solana)
    positions_value = None
    total_unrealized_pnl = None
    if system == "solana" and not positions_df.empty:
        if "value_usd" in positions_df.columns:
            positions_value = float(positions_df["value_usd"].sum())
        if "unrealized_pnl" in positions_df.columns:
            total_unrealized_pnl = float(positions_df["unrealized_pnl"].sum())

    # Prepare all data products
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

    # Upload (or dry-run)
    blobs = {
        f"dashboard/{system}/overview-daily.json": overview_daily,
        f"dashboard/{system}/overview-5min.json": overview_5min,
        f"dashboard/{system}/positions.json": positions_data,
        f"dashboard/{system}/risk-daily.json": risk_daily,
        f"dashboard/{system}/risk-5min.json": risk_5min,
    }

    for path, data in blobs.items():
        if dry_run:
            size = len(json.dumps(data, cls=NumpyEncoder))
            logger.info("[DRY RUN] %s: %d bytes", path, size)
        else:
            _upload_blob(path, data)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Push dashboard data to Vercel Blob")
    parser.add_argument("--system", choices=["oanda", "alpaca", "solana"],
                        help="Push only this system")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute data but don't upload")
    args = parser.parse_args()

    systems = [args.system] if args.system else ["oanda", "alpaca", "solana"]
    for system in systems:
        try:
            push_system(system, dry_run=args.dry_run)
        except Exception:
            logger.exception("Failed to push %s", system)


if __name__ == "__main__":
    main()
```

**Step 5: Run tests**

```bash
cd /home/kingjames/trade-performance-dashboard
python -m pytest tests/test_push_to_vercel.py -v
```

Expected: All 3 tests PASS.

**Step 6: Test dry-run with real data**

```bash
python push_to_vercel.py --dry-run
```

Expected: Logs showing data sizes per system (no upload since no token).

**Step 7: Commit**

```bash
git add push_to_vercel.py tests/test_push_to_vercel.py
git commit -m "feat: add Vercel Blob push script with pre-computed data products"
```

---

## Task 4: SWR Data Hooks

**Files:**
- Create: `web/lib/data.ts`

**Step 1: Create data fetching hooks**

Create `web/lib/data.ts`:

```typescript
"use client";

import useSWR from "swr";
import type { OverviewData, PositionsData, RiskData, SystemSlug, Timeframe } from "./types";

const BLOB_BASE = process.env.NEXT_PUBLIC_BLOB_BASE_URL || "";

const fetcher = (url: string) => fetch(url).then((r) => r.json());

const SWR_OPTIONS = {
  revalidateOnFocus: false,
  refreshInterval: 60_000,
  dedupingInterval: 30_000,
};

export function useOverview(system: SystemSlug, timeframe: Timeframe) {
  const freq = timeframe === "daily" ? "daily" : "5min";
  return useSWR<OverviewData>(
    `${BLOB_BASE}/dashboard/${system}/overview-${freq}.json`,
    fetcher,
    SWR_OPTIONS,
  );
}

export function usePositions(system: SystemSlug) {
  return useSWR<PositionsData>(
    `${BLOB_BASE}/dashboard/${system}/positions.json`,
    fetcher,
    SWR_OPTIONS,
  );
}

export function useRisk(system: SystemSlug, timeframe: Timeframe) {
  const freq = timeframe === "daily" ? "daily" : "5min";
  return useSWR<RiskData>(
    `${BLOB_BASE}/dashboard/${system}/risk-${freq}.json`,
    fetcher,
    SWR_OPTIONS,
  );
}
```

**Step 2: Commit**

```bash
git add web/lib/data.ts
git commit -m "feat: add SWR data hooks for Vercel Blob"
```

---

## Task 5: Root Layout, System Sidebar, and Shell

**Files:**
- Create: `web/components/SystemSidebar.tsx`, `web/components/TimeframeToggle.tsx`, `web/components/MetricCard.tsx`
- Modify: `web/app/layout.tsx`, `web/app/page.tsx`
- Modify: `web/app/globals.css`

**Step 1: Update globals.css for dark theme**

Replace `web/app/globals.css`:

```css
@tailwind base;
@tailwind components;
@tailwind utilities;

:root {
  --bg: #0E1117;
  --card: #1E2130;
}

body {
  background-color: var(--bg);
  color: #FAFAFA;
}
```

**Step 2: Create SystemSidebar**

Create `web/components/SystemSidebar.tsx`:

```tsx
"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { SYSTEMS, SYSTEM_SLUGS, type SystemConfig } from "@/lib/systems";
import type { SystemSlug } from "@/lib/types";

function extractSystem(pathname: string): SystemSlug {
  const segment = pathname.split("/")[1];
  if (SYSTEM_SLUGS.includes(segment as SystemSlug)) return segment as SystemSlug;
  return "oanda";
}

export default function SystemSidebar() {
  const pathname = usePathname();
  const current = extractSystem(pathname);
  const config = SYSTEMS[current];

  return (
    <aside className="w-64 shrink-0 border-r border-white/10 p-5 flex flex-col gap-6">
      <div>
        <h2 className="text-xs font-semibold uppercase tracking-wider text-white/50 mb-3">
          Trading System
        </h2>
        <div className="flex flex-col gap-1">
          {SYSTEM_SLUGS.map((slug) => (
            <Link
              key={slug}
              href={`/${slug}`}
              className={`px-3 py-2 rounded-lg text-sm transition-colors ${
                slug === current
                  ? "bg-primary/20 text-primary font-medium"
                  : "text-white/70 hover:bg-white/5 hover:text-white"
              }`}
            >
              {SYSTEMS[slug].name}
            </Link>
          ))}
        </div>
      </div>

      <div className="border-t border-white/10 pt-4">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-white/50 mb-2">
          System Details
        </h3>
        <ul className="text-sm text-white/60 space-y-1">
          {config.description.map((line, i) => (
            <li key={i}>- {line}</li>
          ))}
        </ul>
        {config.startCaption && (
          <p className="text-xs text-white/40 mt-2">{config.startCaption}</p>
        )}
      </div>
    </aside>
  );
}
```

**Step 3: Create TimeframeToggle**

Create `web/components/TimeframeToggle.tsx`:

```tsx
"use client";

import type { Timeframe } from "@/lib/types";

interface Props {
  value: Timeframe;
  onChange: (tf: Timeframe) => void;
}

export default function TimeframeToggle({ value, onChange }: Props) {
  return (
    <div className="inline-flex rounded-lg bg-card p-1 gap-1">
      {(["daily", "5min"] as const).map((tf) => (
        <button
          key={tf}
          onClick={() => onChange(tf)}
          className={`px-4 py-1.5 rounded-md text-sm font-medium transition-colors ${
            value === tf
              ? "bg-primary text-white"
              : "text-white/60 hover:text-white hover:bg-white/5"
          }`}
        >
          {tf === "daily" ? "Daily" : "5-Minute"}
        </button>
      ))}
    </div>
  );
}
```

**Step 4: Create MetricCard**

Create `web/components/MetricCard.tsx`:

```tsx
interface Props {
  label: string;
  value: string;
}

export default function MetricCard({ label, value }: Props) {
  return (
    <div className="bg-card rounded-xl p-4">
      <p className="text-xs text-white/50 uppercase tracking-wider">{label}</p>
      <p className="text-2xl font-semibold mt-1">{value}</p>
    </div>
  );
}
```

**Step 5: Update root layout**

Replace `web/app/layout.tsx`:

```tsx
import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import SystemSidebar from "@/components/SystemSidebar";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Trading Performance Dashboard",
  description: "Algorithmic trading system metrics",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body className={`${inter.className} min-h-screen flex`}>
        <SystemSidebar />
        <main className="flex-1 overflow-auto">{children}</main>
      </body>
    </html>
  );
}
```

**Step 6: Root redirect**

Replace `web/app/page.tsx`:

```tsx
import { redirect } from "next/navigation";

export default function Home() {
  redirect("/oanda");
}
```

**Step 7: Verify build**

```bash
cd /home/kingjames/trade-performance-dashboard/web
npm run build
```

Expected: Build succeeds (may warn about missing [system] pages — that's OK for now).

**Step 8: Commit**

```bash
git add web/app/ web/components/SystemSidebar.tsx web/components/TimeframeToggle.tsx web/components/MetricCard.tsx
git commit -m "feat: add root layout, system sidebar, timeframe toggle, and metric card"
```

---

## Task 6: Chart Components (Lightweight Charts)

**Files:**
- Create: `web/components/charts/ChartContainer.tsx`, `web/components/charts/EquityCurve.tsx`, `web/components/charts/DrawdownChart.tsx`, `web/components/charts/DailyReturns.tsx`, `web/components/charts/RollingSharpe.tsx`, `web/components/charts/CumulativePnL.tsx`, `web/components/charts/ExposureChart.tsx`

**Step 1: Create ChartContainer (reusable wrapper)**

Create `web/components/charts/ChartContainer.tsx`:

```tsx
"use client";

import { useRef, useEffect } from "react";
import { createChart, type IChartApi, type DeepPartial, type ChartOptions } from "lightweight-charts";

const CHART_DEFAULTS: DeepPartial<ChartOptions> = {
  layout: {
    background: { color: "#0E1117" },
    textColor: "#FAFAFA",
    fontFamily: "Inter, system-ui, sans-serif",
  },
  grid: {
    vertLines: { color: "rgba(255,255,255,0.05)" },
    horzLines: { color: "rgba(255,255,255,0.05)" },
  },
  crosshair: {
    vertLine: { color: "rgba(74,144,217,0.3)" },
    horzLine: { color: "rgba(74,144,217,0.3)" },
  },
  timeScale: {
    borderColor: "rgba(255,255,255,0.1)",
    timeVisible: true,
  },
  rightPriceScale: {
    borderColor: "rgba(255,255,255,0.1)",
  },
};

interface Props {
  options?: DeepPartial<ChartOptions>;
  className?: string;
  onChart: (chart: IChartApi) => void;
}

export default function ChartContainer({ options, className, onChart }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    const chart = createChart(containerRef.current, {
      ...CHART_DEFAULTS,
      ...options,
      width: containerRef.current.clientWidth,
      height: 350,
    });

    chartRef.current = chart;
    onChart(chart);

    const resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        chart.applyOptions({ width: entry.contentRect.width });
      }
    });
    resizeObserver.observe(containerRef.current);

    return () => {
      resizeObserver.disconnect();
      chart.remove();
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  return <div ref={containerRef} className={className ?? "w-full"} />;
}
```

**Step 2: Create EquityCurve chart**

Create `web/components/charts/EquityCurve.tsx`:

```tsx
"use client";

import type { TimeSeries } from "@/lib/types";
import ChartContainer from "./ChartContainer";

interface Props {
  data: TimeSeries;
  title: string;
}

export default function EquityCurve({ data, title }: Props) {
  return (
    <div>
      <h3 className="text-sm font-medium text-white/70 mb-2">{title}</h3>
      <ChartContainer
        onChart={(chart) => {
          const series = chart.addLineSeries({
            color: "#4A90D9",
            lineWidth: 2,
            priceFormat: { type: "price", precision: 2, minMove: 0.01 },
          });
          series.setData(
            data.timestamps.map((t, i) => ({
              time: t.slice(0, 10) as `${number}-${number}-${number}`,
              value: data.values[i],
            }))
          );
          chart.timeScale().fitContent();
        }}
      />
    </div>
  );
}
```

**Step 3: Create DrawdownChart**

Create `web/components/charts/DrawdownChart.tsx`:

```tsx
"use client";

import type { TimeSeries } from "@/lib/types";
import ChartContainer from "./ChartContainer";

interface Props {
  data: TimeSeries;
}

export default function DrawdownChart({ data }: Props) {
  return (
    <div>
      <h3 className="text-sm font-medium text-white/70 mb-2">Drawdown</h3>
      <ChartContainer
        onChart={(chart) => {
          const series = chart.addAreaSeries({
            lineColor: "#FF1744",
            topColor: "rgba(255, 23, 68, 0.4)",
            bottomColor: "rgba(255, 23, 68, 0.0)",
            lineWidth: 1,
            priceFormat: { type: "percent" },
          });
          series.setData(
            data.timestamps.map((t, i) => ({
              time: t.slice(0, 10) as `${number}-${number}-${number}`,
              value: data.values[i] * 100,
            }))
          );
          chart.timeScale().fitContent();
        }}
      />
    </div>
  );
}
```

**Step 4: Create DailyReturns histogram**

Create `web/components/charts/DailyReturns.tsx`:

```tsx
"use client";

import type { TimeSeries } from "@/lib/types";
import ChartContainer from "./ChartContainer";

interface Props {
  data: TimeSeries;
}

export default function DailyReturns({ data }: Props) {
  return (
    <div>
      <h3 className="text-sm font-medium text-white/70 mb-2">Daily Returns</h3>
      <ChartContainer
        onChart={(chart) => {
          const series = chart.addHistogramSeries({
            priceFormat: { type: "percent" },
          });
          series.setData(
            data.timestamps.map((t, i) => ({
              time: t.slice(0, 10) as `${number}-${number}-${number}`,
              value: data.values[i] * 100,
              color: data.values[i] >= 0 ? "#00C853" : "#FF1744",
            }))
          );
          chart.timeScale().fitContent();
        }}
      />
    </div>
  );
}
```

**Step 5: Create RollingSharpe**

Create `web/components/charts/RollingSharpe.tsx`:

```tsx
"use client";

import type { TimeSeries } from "@/lib/types";
import ChartContainer from "./ChartContainer";

interface Props {
  data: TimeSeries;
  window: number;
}

export default function RollingSharpe({ data, window }: Props) {
  if (data.timestamps.length === 0) return null;

  return (
    <div>
      <h3 className="text-sm font-medium text-white/70 mb-2">
        Rolling Sharpe ({window}-Period)
      </h3>
      <ChartContainer
        onChart={(chart) => {
          const series = chart.addLineSeries({
            color: "#AA00FF",
            lineWidth: 2,
          });
          series.setData(
            data.timestamps.map((t, i) => ({
              time: t.slice(0, 10) as `${number}-${number}-${number}`,
              value: data.values[i],
            }))
          );
          chart.timeScale().fitContent();
        }}
      />
    </div>
  );
}
```

**Step 6: Create CumulativePnL multi-line chart**

Create `web/components/charts/CumulativePnL.tsx`:

```tsx
"use client";

import ChartContainer from "./ChartContainer";

const PALETTE = [
  "#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854",
  "#ffd92f", "#e5c494", "#b3b3b3", "#e41a1c", "#377eb8",
  "#4daf4a", "#984ea3", "#ff7f00", "#ffff33", "#a65628",
  "#f781bf", "#999999", "#66c2a5", "#fc8d62", "#8da0cb",
];

interface Props {
  timestamps: string[];
  series: Record<string, number[]>;
  title: string;
}

export default function CumulativePnL({ timestamps, series, title }: Props) {
  const symbols = Object.keys(series);
  if (symbols.length === 0) return null;

  return (
    <div>
      <h3 className="text-sm font-medium text-white/70 mb-2">{title}</h3>
      <ChartContainer
        options={{ height: 450 } as any}
        onChart={(chart) => {
          symbols.forEach((sym, i) => {
            const line = chart.addLineSeries({
              color: PALETTE[i % PALETTE.length],
              lineWidth: 2,
              title: sym,
            });
            line.setData(
              timestamps.map((t, j) => ({
                time: t.slice(0, 10) as `${number}-${number}-${number}`,
                value: series[sym][j] ?? 0,
              }))
            );
          });
          chart.timeScale().fitContent();
        }}
      />
    </div>
  );
}
```

**Step 7: Create ExposureChart**

Create `web/components/charts/ExposureChart.tsx`:

```tsx
"use client";

import ChartContainer from "./ChartContainer";

const EXPOSURE_COLORS: Record<string, { line: string; top: string }> = {
  "Long Exposure":    { line: "#00C853", top: "rgba(0, 200, 83, 0.3)" },
  "Short Exposure":   { line: "#FF1744", top: "rgba(255, 23, 68, 0.3)" },
  "Margin Used":      { line: "#FF9100", top: "rgba(255, 145, 0, 0.3)" },
  "Token Positions":  { line: "#00C853", top: "rgba(0, 200, 83, 0.3)" },
  "SOL Balance":      { line: "#4A90D9", top: "rgba(74, 144, 217, 0.3)" },
};

interface Props {
  timestamps: string[];
  series: Record<string, number[]>;
}

export default function ExposureChart({ timestamps, series }: Props) {
  const labels = Object.keys(series);
  if (labels.length === 0) return null;

  return (
    <div>
      <h3 className="text-sm font-medium text-white/70 mb-2">Exposure Over Time</h3>
      <ChartContainer
        onChart={(chart) => {
          labels.forEach((label) => {
            const colors = EXPOSURE_COLORS[label] ?? { line: "#4A90D9", top: "rgba(74,144,217,0.3)" };
            const area = chart.addAreaSeries({
              lineColor: colors.line,
              topColor: colors.top,
              bottomColor: "transparent",
              lineWidth: 1,
              title: label,
            });
            area.setData(
              timestamps.map((t, i) => ({
                time: t.slice(0, 10) as `${number}-${number}-${number}`,
                value: series[label][i] ?? 0,
              }))
            );
          });
          chart.timeScale().fitContent();
        }}
      />
    </div>
  );
}
```

**Step 8: Commit**

```bash
git add web/components/charts/
git commit -m "feat: add Lightweight Charts components (equity, drawdown, returns, sharpe, pnl, exposure)"
```

---

## Task 7: HorizontalBar and DataTable Components

**Files:**
- Create: `web/components/HorizontalBar.tsx`, `web/components/DataTable.tsx`

**Step 1: Create HorizontalBar**

Create `web/components/HorizontalBar.tsx`:

```tsx
interface BarItem {
  label: string;
  value: number;
}

interface Props {
  items: BarItem[];
  title: string;
  formatValue?: (v: number) => string;
}

export default function HorizontalBar({ items, title, formatValue }: Props) {
  if (items.length === 0) return null;

  const maxAbs = Math.max(...items.map((d) => Math.abs(d.value)), 0.01);
  const fmt = formatValue ?? ((v) => `$${v.toFixed(2)}`);

  return (
    <div>
      <h3 className="text-sm font-medium text-white/70 mb-3">{title}</h3>
      <div
        className="space-y-1 overflow-y-auto"
        style={{ maxHeight: Math.max(400, items.length * 28) }}
      >
        {items.map((item) => {
          const pct = (Math.abs(item.value) / maxAbs) * 100;
          const isPositive = item.value >= 0;
          return (
            <div key={item.label} className="flex items-center gap-2 text-xs">
              <span className="w-20 text-right text-white/60 shrink-0 truncate">
                {item.label}
              </span>
              <div className="flex-1 h-5 bg-white/5 rounded relative">
                <div
                  className={`h-full rounded ${isPositive ? "bg-accent-green/70" : "bg-accent-red/70"}`}
                  style={{ width: `${pct}%` }}
                />
              </div>
              <span
                className={`w-20 text-right shrink-0 ${
                  isPositive ? "text-accent-green" : "text-accent-red"
                }`}
              >
                {fmt(item.value)}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
```

**Step 2: Create DataTable**

Create `web/components/DataTable.tsx`:

```tsx
interface Column {
  key: string;
  label: string;
  format?: (v: any) => string;
  align?: "left" | "right";
}

interface Props {
  columns: Column[];
  rows: Record<string, any>[];
  title?: string;
}

export default function DataTable({ columns, rows, title }: Props) {
  if (rows.length === 0) return null;

  return (
    <div>
      {title && (
        <h3 className="text-sm font-medium text-white/70 mb-2">{title}</h3>
      )}
      <div className="overflow-x-auto rounded-lg border border-white/10">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-white/10 bg-card">
              {columns.map((col) => (
                <th
                  key={col.key}
                  className={`px-3 py-2 font-medium text-white/50 ${
                    col.align === "right" ? "text-right" : "text-left"
                  }`}
                >
                  {col.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => (
              <tr key={i} className="border-b border-white/5 hover:bg-white/5">
                {columns.map((col) => (
                  <td
                    key={col.key}
                    className={`px-3 py-1.5 ${
                      col.align === "right" ? "text-right" : "text-left"
                    }`}
                  >
                    {col.format ? col.format(row[col.key]) : row[col.key]}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
```

**Step 3: Commit**

```bash
git add web/components/HorizontalBar.tsx web/components/DataTable.tsx
git commit -m "feat: add HorizontalBar and DataTable components"
```

---

## Task 8: System Layout with Tab Navigation

**Files:**
- Create: `web/app/[system]/layout.tsx`

**Step 1: Create system layout with tabs**

Create `web/app/[system]/layout.tsx`:

```tsx
"use client";

import Link from "next/link";
import { usePathname, useParams } from "next/navigation";
import { useState } from "react";
import TimeframeToggle from "@/components/TimeframeToggle";
import { SYSTEMS } from "@/lib/systems";
import type { SystemSlug, Timeframe } from "@/lib/types";

const TABS = [
  { label: "Overview", href: "" },
  { label: "Positions", href: "/positions" },
  { label: "Risk", href: "/risk" },
];

export default function SystemLayout({ children }: { children: React.ReactNode }) {
  const params = useParams();
  const pathname = usePathname();
  const system = params.system as SystemSlug;
  const config = SYSTEMS[system];
  const [timeframe, setTimeframe] = useState<Timeframe>("daily");

  if (!config) return <div className="p-8">Unknown system</div>;

  // Determine active tab
  const basePath = `/${system}`;

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Algorithmic Trading Performance</h1>
          <p className="text-sm text-white/50">Live system metrics | Updated every 5 minutes</p>
        </div>
        <TimeframeToggle value={timeframe} onChange={setTimeframe} />
      </div>

      <nav className="flex gap-1 border-b border-white/10 pb-px">
        {TABS.map((tab) => {
          const href = `${basePath}${tab.href}`;
          const isActive =
            tab.href === ""
              ? pathname === basePath || pathname === `${basePath}/`
              : pathname.startsWith(href);
          return (
            <Link
              key={tab.label}
              href={href}
              className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
                isActive
                  ? "border-primary text-primary"
                  : "border-transparent text-white/60 hover:text-white"
              }`}
            >
              {tab.label}
            </Link>
          );
        })}
      </nav>

      <TimeframeContext.Provider value={timeframe}>
        {children}
      </TimeframeContext.Provider>
    </div>
  );
}

// Context for child pages to access timeframe without prop drilling
import { createContext, useContext } from "react";

export const TimeframeContext = createContext<Timeframe>("daily");
export function useTimeframe() {
  return useContext(TimeframeContext);
}
```

**Step 2: Commit**

```bash
git add web/app/\[system\]/layout.tsx
git commit -m "feat: add system layout with tab navigation and timeframe toggle"
```

---

## Task 9: Overview Page

**Files:**
- Create: `web/app/[system]/page.tsx`

**Step 1: Create overview page**

Create `web/app/[system]/page.tsx`:

```tsx
"use client";

import { useParams } from "next/navigation";
import { useOverview } from "@/lib/data";
import { useTimeframe } from "./layout";
import MetricCard from "@/components/MetricCard";
import EquityCurve from "@/components/charts/EquityCurve";
import DailyReturns from "@/components/charts/DailyReturns";
import { SYSTEMS } from "@/lib/systems";
import { fmtPct, fmtNumber } from "@/lib/format";
import type { SystemSlug } from "@/lib/types";

export default function OverviewPage() {
  const { system } = useParams() as { system: SystemSlug };
  const timeframe = useTimeframe();
  const { data, isLoading, error } = useOverview(system, timeframe);
  const config = SYSTEMS[system];

  if (isLoading) {
    return <div className="text-white/50 animate-pulse">Loading overview...</div>;
  }
  if (error || !data) {
    return <div className="text-accent-red">Failed to load data</div>;
  }

  const m = data.metrics;
  const periodLabel = timeframe === "5min" ? "Periods" : "Trading Days";
  const curveLabel = `${config.valueName} (${timeframe === "daily" ? "Daily" : "5-Minute"})`;

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
        <MetricCard label="Total Return" value={fmtPct(m.total_return)} />
        <MetricCard label="Sharpe Ratio" value={fmtNumber(m.sharpe)} />
        <MetricCard label="Sortino Ratio" value={fmtNumber(m.sortino)} />
        <MetricCard label="Max Drawdown" value={fmtPct(m.max_drawdown)} />
        <MetricCard label="Win Rate" value={fmtPct(m.win_rate, 1)} />
        <MetricCard label={periodLabel} value={String(m.trading_days)} />
      </div>

      <EquityCurve data={data.equity_curve} title={`${curveLabel} Curve`} />
      <DailyReturns data={data.daily_returns} />
    </div>
  );
}
```

**Step 2: Commit**

```bash
git add web/app/\[system\]/page.tsx
git commit -m "feat: add overview page with metrics, equity curve, and daily returns"
```

---

## Task 10: Positions Page

**Files:**
- Create: `web/app/[system]/positions/page.tsx`

**Step 1: Create positions page**

Create `web/app/[system]/positions/page.tsx`:

```tsx
"use client";

import { useParams } from "next/navigation";
import { usePositions } from "@/lib/data";
import { useTimeframe } from "../layout";
import MetricCard from "@/components/MetricCard";
import HorizontalBar from "@/components/HorizontalBar";
import CumulativePnL from "@/components/charts/CumulativePnL";
import DataTable from "@/components/DataTable";
import { fmtDollar, fmtPct, fmtNumber, fmtSignedDollar } from "@/lib/format";
import type { SystemSlug, InstrumentMetric } from "@/lib/types";

export default function PositionsPage() {
  const { system } = useParams() as { system: SystemSlug };
  const timeframe = useTimeframe();
  const { data, isLoading, error } = usePositions(system);

  if (isLoading) {
    return <div className="text-white/50 animate-pulse">Loading positions...</div>;
  }
  if (error || !data) {
    return <div className="text-accent-red">Failed to load data</div>;
  }

  const freq = timeframe === "daily" ? "daily" : "five_min";
  const instMetrics: InstrumentMetric[] = data.instrument_metrics[freq];
  const cumPnl = data.cumulative_pnl[freq];
  const pnlLabel = timeframe === "5min" ? "Avg 5-Min P&L" : "Avg Daily P&L";
  const periodsLabel = timeframe === "5min" ? "Periods" : "Days";

  // Position summary metrics (system-specific)
  const summaryCards = [];
  if (system === "oanda") {
    summaryCards.push(
      { label: "Open Positions", value: String(data.summary.total) },
    );
    if (data.summary.algo_pnl != null) {
      summaryCards.push({ label: "Algorithm P&L", value: fmtSignedDollar(data.summary.algo_pnl) });
    }
  } else if (system === "alpaca") {
    summaryCards.push(
      { label: "Total Positions", value: String(data.summary.total) },
      { label: "Long", value: String(data.summary.long) },
      { label: "Short", value: String(data.summary.short) },
    );
  } else {
    summaryCards.push(
      { label: "Open Positions", value: String(data.summary.total) },
    );
    if (data.summary.positions_value != null) {
      summaryCards.push({ label: "Positions Value", value: fmtDollar(data.summary.positions_value) });
    }
    if (data.summary.total_unrealized_pnl != null) {
      summaryCards.push({ label: "Unrealized P&L", value: fmtSignedDollar(data.summary.total_unrealized_pnl) });
    }
  }

  // P&L field varies by system
  const plField = system === "oanda" ? "unrealized_pl" : "unrealized_pl";
  const symbolField = system === "oanda" ? "instrument" : "symbol";

  // Sort positions for bar chart
  const sortedPositions = [...data.positions].sort(
    (a, b) => (a[plField] as number) - (b[plField] as number)
  );

  // Instrument metrics table columns
  const metricCols = [
    { key: "instrument", label: system === "oanda" ? "Instrument" : "Symbol" },
    { key: "total_pnl", label: "Total P&L", align: "right" as const, format: (v: number) => fmtDollar(v) },
    { key: "avg_daily_pnl", label: pnlLabel, align: "right" as const, format: (v: number) => fmtDollar(v) },
    { key: "sharpe", label: "Sharpe", align: "right" as const, format: (v: number) => fmtNumber(v) },
    { key: "sortino", label: "Sortino", align: "right" as const, format: (v: number) => fmtNumber(v) },
    { key: "win_rate", label: "Win Rate", align: "right" as const, format: (v: number) => fmtPct(v, 1) },
    { key: "trading_days", label: periodsLabel, align: "right" as const },
  ];

  return (
    <div className="space-y-8">
      {/* Position summary */}
      <div>
        <h2 className="text-lg font-semibold mb-3">Current Positions</h2>
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
          {summaryCards.map((c) => (
            <MetricCard key={c.label} label={c.label} value={c.value} />
          ))}
        </div>
      </div>

      {/* Cumulative P&L chart */}
      {cumPnl.timestamps.length > 0 && (
        <CumulativePnL
          timestamps={cumPnl.timestamps}
          series={cumPnl.series}
          title={`Cumulative P&L — ${system === "alpaca" ? "Top/Bottom 10" : "By Instrument"} (${timeframe === "daily" ? "Daily" : "5-Minute"})`}
        />
      )}

      {/* Unrealized P&L bar chart */}
      <HorizontalBar
        items={sortedPositions.map((p) => ({
          label: p[symbolField] as string,
          value: p[plField] as number,
        }))}
        title={system === "oanda" ? "Unrealized P&L by Instrument" : "Unrealized P&L by Symbol"}
      />

      {/* Per-instrument performance */}
      <div>
        <h2 className="text-lg font-semibold mb-3">Per-Instrument Performance</h2>

        {/* Sharpe bar chart */}
        <HorizontalBar
          items={[...instMetrics]
            .sort((a, b) => a.sharpe - b.sharpe)
            .map((m) => ({ label: m.instrument, value: m.sharpe }))}
          title={`Per-Instrument Sharpe (${timeframe === "daily" ? "Daily" : "5-Minute"})`}
          formatValue={(v) => v.toFixed(2)}
        />

        {/* Metrics table */}
        <div className="mt-6">
          <DataTable columns={metricCols} rows={instMetrics} />
        </div>
      </div>
    </div>
  );
}
```

**Step 2: Commit**

```bash
git add web/app/\[system\]/positions/
git commit -m "feat: add positions page with P&L charts, instrument metrics, and data table"
```

---

## Task 11: Risk Page

**Files:**
- Create: `web/app/[system]/risk/page.tsx`

**Step 1: Create risk page**

Create `web/app/[system]/risk/page.tsx`:

```tsx
"use client";

import { useParams } from "next/navigation";
import { useRisk } from "@/lib/data";
import { useTimeframe } from "../layout";
import MetricCard from "@/components/MetricCard";
import DrawdownChart from "@/components/charts/DrawdownChart";
import RollingSharpe from "@/components/charts/RollingSharpe";
import ExposureChart from "@/components/charts/ExposureChart";
import { fmtPct, fmtNumber } from "@/lib/format";
import type { SystemSlug } from "@/lib/types";

export default function RiskPage() {
  const { system } = useParams() as { system: SystemSlug };
  const timeframe = useTimeframe();
  const { data, isLoading, error } = useRisk(system, timeframe);

  if (isLoading) {
    return <div className="text-white/50 animate-pulse">Loading risk data...</div>;
  }
  if (error || !data) {
    return <div className="text-accent-red">Failed to load data</div>;
  }

  const m = data.risk_metrics;
  const retLabel = timeframe === "5min" ? "Avg 5-Min Return" : "Avg Daily Return";
  const volLabel = timeframe === "5min" ? "5-Min Volatility" : "Daily Volatility";
  const rollingWindow = timeframe === "5min" ? 288 : 10;

  return (
    <div className="space-y-6">
      <DrawdownChart data={data.drawdown} />
      <RollingSharpe data={data.rolling_sharpe} window={rollingWindow} />
      <ExposureChart timestamps={data.exposure.timestamps} series={data.exposure.series} />

      <div>
        <h2 className="text-lg font-semibold mb-3">Risk Summary</h2>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <MetricCard label="Calmar Ratio" value={fmtNumber(m.calmar)} />
          <MetricCard label="Annualized Return" value={fmtPct(m.annualized_return)} />
          <MetricCard label={retLabel} value={fmtPct(m.avg_return, 4)} />
          <MetricCard label={volLabel} value={fmtPct(m.volatility, 4)} />
        </div>
      </div>
    </div>
  );
}
```

**Step 2: Verify full build**

```bash
cd /home/kingjames/trade-performance-dashboard/web
npm run build
```

Expected: Build succeeds.

**Step 3: Commit**

```bash
git add web/app/\[system\]/risk/
git commit -m "feat: add risk page with drawdown, rolling sharpe, exposure, and risk metrics"
```

---

## Task 12: Vercel Deployment Setup

**Files:**
- Create: `web/.env.example`
- Modify: `web/package.json` (if needed)

**Step 1: Create env example**

Create `web/.env.example`:

```
NEXT_PUBLIC_BLOB_BASE_URL=https://your-blob-store.public.blob.vercel-storage.com
```

**Step 2: Create Vercel Blob store**

```bash
cd /home/kingjames/trade-performance-dashboard/web
npx vercel link   # link to Vercel project
npx vercel blob create-store dashboard-data
```

Note the `BLOB_READ_WRITE_TOKEN` value for the push script.

**Step 3: Set env vars on Vercel**

```bash
npx vercel env add NEXT_PUBLIC_BLOB_BASE_URL
# Enter the blob store public URL
```

**Step 4: Set env var on home server for push script**

```bash
# Add to the systemd service environment
echo 'BLOB_READ_WRITE_TOKEN=vercel_blob_rw_...' >> ~/.env.dashboard
```

**Step 5: Initial data push**

```bash
cd /home/kingjames/trade-performance-dashboard
source ~/.env.dashboard
python push_to_vercel.py
```

Expected: Logs showing successful uploads for all 3 systems.

**Step 6: Deploy to Vercel**

```bash
cd /home/kingjames/trade-performance-dashboard/web
npx vercel --prod
```

Expected: Deployment succeeds, accessible at the Vercel URL.

**Step 7: Commit**

```bash
git add web/.env.example
git commit -m "feat: add Vercel deployment config"
```

---

## Task 13: Update Systemd Timer to Push Data

**Files:**
- Modify: `update_cache.py`
- Modify: `systemd/trade-cache-refresh.service`

**Step 1: Read current update_cache.py**

Check the current contents and add push_to_vercel call at the end.

**Step 2: Modify update_cache.py to also push**

Add to the end of `update_cache.py`:

```python
# After cache refresh, push to Vercel Blob
try:
    from push_to_vercel import push_system
    for system in ["oanda", "alpaca", "solana"]:
        push_system(system)
except Exception as e:
    logger.warning("Vercel push failed (non-fatal): %s", e)
```

**Step 3: Add BLOB_READ_WRITE_TOKEN to systemd service**

Modify `systemd/trade-cache-refresh.service` to add:

```ini
EnvironmentFile=/home/kingjames/.env.dashboard
```

**Step 4: Reload and restart timer**

```bash
sudo systemctl daemon-reload
sudo systemctl restart trade-cache-refresh.timer
```

**Step 5: Verify timer fires and pushes**

```bash
sudo systemctl start trade-cache-refresh.service
journalctl -u trade-cache-refresh.service -n 20
```

Expected: Logs showing cache refresh followed by Vercel Blob uploads.

**Step 6: Commit**

```bash
git add update_cache.py systemd/
git commit -m "feat: integrate Vercel Blob push into systemd cache refresh timer"
```

---

## Verification Checklist

After all tasks are complete:

1. [ ] `npm run build` succeeds in `web/`
2. [ ] `python -m pytest tests/test_push_to_vercel.py` passes
3. [ ] `python push_to_vercel.py --dry-run` shows correct data sizes
4. [ ] Vercel deployment is live and loads OANDA data
5. [ ] System switching (OANDA → Alpaca → Solana) is instant
6. [ ] Tab switching (Overview → Positions → Risk) is instant
7. [ ] Timeframe toggle (Daily ↔ 5-Minute) updates within 1 second
8. [ ] All charts render correctly
9. [ ] Systemd timer pushes data every 5 minutes
10. [ ] Data revalidates silently in the browser (SWR 60s)
