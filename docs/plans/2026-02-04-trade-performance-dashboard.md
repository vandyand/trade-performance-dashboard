# Trade Performance Dashboard Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a professional Streamlit dashboard showcasing live algorithmic trading performance from OANDA (forex) and Alpaca (equities) systems, with verifiable metrics for client pitches and Upwork proposals.

**Architecture:** Single-page Streamlit app with tabbed layout (Overview | Positions | Risk). Reads JSONL log files produced by existing systemd samplers running on the same machine. Separate modules for data loading (`data_loader.py`), metrics computation (`metrics.py`), and chart creation (`charts.py`). Sidebar controls system selection, date range, and phase filtering.

**Tech Stack:** Python 3.12, Streamlit >= 1.30, Pandas >= 2.0, Plotly >= 5.18, NumPy >= 1.26

**Data Sources:**
- OANDA: `/home/kingjames/rl-trader/forex-rl/logs/oanda_nav_positions.jsonl` (~10K records, 5-min intervals, Dec 30 2025 - present)
- Alpaca: `/home/kingjames/rl-trader/forex-rl/logs/alpaca_equity_positions.jsonl` (~600+ records, 5-min intervals, Feb 2 2026 - present)

**Project Location:** `/home/kingjames/trade-performance-dashboard/`

---

### Task 1: Project Scaffolding

**Files:**
- Create: `requirements.txt`
- Create: `.streamlit/config.toml`
- Create: `tests/__init__.py`

**Step 1: Create project directory structure**

```bash
mkdir -p /home/kingjames/trade-performance-dashboard/{tests,.streamlit}
cd /home/kingjames/trade-performance-dashboard
python3 -m venv venv
```

**Step 2: Create requirements.txt**

```
streamlit>=1.30.0
pandas>=2.0.0
plotly>=5.18.0
numpy>=1.26.0
pytest>=7.0.0
```

**Step 3: Install dependencies**

```bash
source venv/bin/activate && pip install -r requirements.txt
```

**Step 4: Create .streamlit/config.toml**

```toml
[theme]
primaryColor = "#4A90D9"
backgroundColor = "#0E1117"
secondaryBackgroundColor = "#1E2130"
textColor = "#FAFAFA"
```

**Step 5: Create empty tests/__init__.py**

```bash
touch tests/__init__.py
```

**Step 6: Initialize git and commit**

```bash
git init
git add .
git commit -m "chore: project scaffolding"
```

---

### Task 2: Data Loader Module

**Files:**
- Create: `tests/test_data_loader.py`
- Create: `data_loader.py`

**Step 1: Write failing tests for data loader**

```python
# tests/test_data_loader.py
import json
import pytest
import pandas as pd
from pathlib import Path

from data_loader import load_oanda_nav, load_alpaca_equity, resample_daily


@pytest.fixture
def oanda_jsonl(tmp_path):
    """Two OANDA snapshots 5 minutes apart."""
    records = [
        {
            "event": "oanda_nav_positions_snapshot",
            "time_iso": "2026-01-19T10:00:00+00:00",
            "epoch": 1737280800.0,
            "account": {
                "NAV": "980.00", "balance": "978.00",
                "pl": "-22.00", "unrealizedPL": "2.00",
                "financing": "-5.00", "openTradeCount": 5,
                "marginUsed": "50.00", "positionValue": "1200.00"
            },
            "positions": [
                {
                    "instrument": "EUR_USD",
                    "long": {"units": "100", "pl": "5.00", "unrealizedPL": "1.00"},
                    "short": {"units": "0", "pl": "-2.00", "unrealizedPL": "0.00"},
                    "pl": "3.00", "unrealizedPL": "1.00"
                }
            ]
        },
        {
            "event": "oanda_nav_positions_snapshot",
            "time_iso": "2026-01-19T10:05:00+00:00",
            "epoch": 1737281100.0,
            "account": {
                "NAV": "981.50", "balance": "978.00",
                "pl": "-22.00", "unrealizedPL": "3.50",
                "financing": "-5.00", "openTradeCount": 5,
                "marginUsed": "51.00", "positionValue": "1210.00"
            },
            "positions": [
                {
                    "instrument": "EUR_USD",
                    "long": {"units": "100", "pl": "5.50", "unrealizedPL": "1.50"},
                    "short": {"units": "0", "pl": "-2.00", "unrealizedPL": "0.00"},
                    "pl": "3.50", "unrealizedPL": "1.50"
                }
            ]
        },
    ]
    f = tmp_path / "oanda.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return f


@pytest.fixture
def alpaca_jsonl(tmp_path):
    """Two Alpaca snapshots 5 minutes apart."""
    records = [
        {
            "event": "alpaca_equity_snapshot",
            "time_iso": "2026-02-03T15:00:00+00:00",
            "epoch": 1738594800.0,
            "paper": True,
            "account": {
                "equity": 100000.0, "last_equity": 99500.0,
                "cash": 80000.0, "long_market_value": 15000.0,
                "short_market_value": -5000.0,
                "equity_change_today": 500.0,
                "equity_change_today_pct": 0.005
            },
            "positions": [
                {
                    "symbol": "AAPL", "qty": 10.0, "side": "PositionSide.LONG",
                    "market_value": 2000.0, "cost_basis": 1900.0,
                    "avg_entry_price": 190.0,
                    "unrealized_pl": 100.0, "unrealized_plpc": 0.0526
                }
            ],
            "totals": {
                "positions_count": 1, "positions_market_value": 2000.0,
                "positions_unrealized_pl": 100.0
            }
        },
        {
            "event": "alpaca_equity_snapshot",
            "time_iso": "2026-02-03T15:05:00+00:00",
            "epoch": 1738595100.0,
            "paper": True,
            "account": {
                "equity": 100050.0, "last_equity": 99500.0,
                "cash": 80000.0, "long_market_value": 15050.0,
                "short_market_value": -5000.0,
                "equity_change_today": 550.0,
                "equity_change_today_pct": 0.0055
            },
            "positions": [
                {
                    "symbol": "AAPL", "qty": 10.0, "side": "PositionSide.LONG",
                    "market_value": 2050.0, "cost_basis": 1900.0,
                    "avg_entry_price": 190.0,
                    "unrealized_pl": 150.0, "unrealized_plpc": 0.0789
                }
            ],
            "totals": {
                "positions_count": 1, "positions_market_value": 2050.0,
                "positions_unrealized_pl": 150.0
            }
        },
    ]
    f = tmp_path / "alpaca.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return f


class TestLoadOandaNav:
    def test_returns_dataframe_with_expected_columns(self, oanda_jsonl):
        df = load_oanda_nav(oanda_jsonl)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        for col in ["nav", "balance", "pl", "unrealized_pl", "financing",
                     "open_trade_count", "margin_used", "position_value"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_nav_values_are_float(self, oanda_jsonl):
        df = load_oanda_nav(oanda_jsonl)
        assert df["nav"].iloc[0] == 980.00
        assert df["nav"].iloc[1] == 981.50

    def test_has_datetime_index(self, oanda_jsonl):
        df = load_oanda_nav(oanda_jsonl)
        assert df.index.name == "timestamp"
        assert pd.api.types.is_datetime64_any_dtype(df.index)

    def test_returns_empty_dataframe_for_missing_file(self, tmp_path):
        df = load_oanda_nav(tmp_path / "nonexistent.jsonl")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0


class TestLoadAlpacaEquity:
    def test_returns_dataframe_with_expected_columns(self, alpaca_jsonl):
        df = load_alpaca_equity(alpaca_jsonl)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        for col in ["equity", "cash", "long_market_value",
                     "short_market_value", "positions_count"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_equity_values_are_float(self, alpaca_jsonl):
        df = load_alpaca_equity(alpaca_jsonl)
        assert df["equity"].iloc[0] == 100000.0
        assert df["equity"].iloc[1] == 100050.0

    def test_has_datetime_index(self, alpaca_jsonl):
        df = load_alpaca_equity(alpaca_jsonl)
        assert df.index.name == "timestamp"


class TestResampleDaily:
    def test_resamples_to_last_value_per_day(self, oanda_jsonl):
        df = load_oanda_nav(oanda_jsonl)
        daily = resample_daily(df, value_col="nav")
        # Both records are same day, so should collapse to 1
        assert len(daily) == 1
        assert daily["nav"].iloc[0] == 981.50  # last value of the day
```

**Step 2: Run tests to verify they fail**

Run: `venv/bin/pytest tests/test_data_loader.py -v`
Expected: ImportError - `data_loader` module doesn't exist.

**Step 3: Write minimal data_loader.py**

```python
# data_loader.py
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
```

**Step 4: Run tests to verify they pass**

Run: `venv/bin/pytest tests/test_data_loader.py -v`
Expected: All pass.

**Step 5: Commit**

```bash
git add data_loader.py tests/test_data_loader.py
git commit -m "feat: data loader for OANDA and Alpaca JSONL files"
```

---

### Task 3: Positions Data Loader

**Files:**
- Modify: `tests/test_data_loader.py` (add position tests)
- Modify: `data_loader.py` (add position extraction functions)

**Step 1: Write failing tests for position extraction**

Append to `tests/test_data_loader.py`:

```python
from data_loader import extract_oanda_positions, extract_alpaca_positions


class TestExtractOandaPositions:
    def test_returns_position_dataframe(self, oanda_jsonl):
        positions = extract_oanda_positions(oanda_jsonl)
        assert isinstance(positions, pd.DataFrame)
        assert "instrument" in positions.columns
        assert "pl" in positions.columns
        assert "unrealized_pl" in positions.columns
        assert "net_units" in positions.columns

    def test_extracts_from_latest_snapshot(self, oanda_jsonl):
        positions = extract_oanda_positions(oanda_jsonl)
        assert len(positions) == 1
        assert positions.iloc[0]["instrument"] == "EUR_USD"
        assert positions.iloc[0]["pl"] == 3.50  # from second (latest) snapshot


class TestExtractAlpacaPositions:
    def test_returns_position_dataframe(self, alpaca_jsonl):
        positions = extract_alpaca_positions(alpaca_jsonl)
        assert isinstance(positions, pd.DataFrame)
        assert "symbol" in positions.columns
        assert "unrealized_pl" in positions.columns
        assert "market_value" in positions.columns

    def test_extracts_from_latest_snapshot(self, alpaca_jsonl):
        positions = extract_alpaca_positions(alpaca_jsonl)
        assert len(positions) == 1
        assert positions.iloc[0]["symbol"] == "AAPL"
        assert positions.iloc[0]["unrealized_pl"] == 150.0
```

**Step 2: Run tests to verify they fail**

Run: `venv/bin/pytest tests/test_data_loader.py -v -k "Position"`
Expected: ImportError - functions don't exist.

**Step 3: Add position extraction to data_loader.py**

Append to `data_loader.py`:

```python
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
```

**Step 4: Run tests to verify they pass**

Run: `venv/bin/pytest tests/test_data_loader.py -v`
Expected: All pass.

**Step 5: Commit**

```bash
git add data_loader.py tests/test_data_loader.py
git commit -m "feat: position extraction from latest OANDA/Alpaca snapshots"
```

---

### Task 4: Metrics Calculator Module

**Files:**
- Create: `tests/test_metrics.py`
- Create: `metrics.py`

**Step 1: Write failing tests for metrics**

```python
# tests/test_metrics.py
import numpy as np
import pandas as pd
import pytest

from metrics import (
    compute_daily_returns,
    compute_sharpe,
    compute_sortino,
    compute_max_drawdown,
    compute_total_return,
    compute_win_rate,
    compute_calmar,
    compute_all_metrics,
)


@pytest.fixture
def daily_nav():
    """15 days of NAV data with known properties."""
    dates = pd.date_range("2026-01-19", periods=15, freq="D")
    # Steady uptrend with small dips
    navs = [980, 982, 981, 984, 986, 985, 988, 990, 989, 992,
            994, 993, 996, 998, 1000]
    return pd.DataFrame({"nav": navs}, index=dates)


@pytest.fixture
def flat_nav():
    """Flat NAV - zero returns."""
    dates = pd.date_range("2026-01-19", periods=5, freq="D")
    return pd.DataFrame({"nav": [1000, 1000, 1000, 1000, 1000]}, index=dates)


class TestComputeDailyReturns:
    def test_returns_series_of_correct_length(self, daily_nav):
        returns = compute_daily_returns(daily_nav, "nav")
        assert len(returns) == 14  # one less than input

    def test_first_return_is_correct(self, daily_nav):
        returns = compute_daily_returns(daily_nav, "nav")
        expected = (982 - 980) / 980
        assert abs(returns.iloc[0] - expected) < 1e-10


class TestComputeSharpe:
    def test_positive_for_uptrend(self, daily_nav):
        returns = compute_daily_returns(daily_nav, "nav")
        sharpe = compute_sharpe(returns)
        assert sharpe > 0

    def test_zero_for_flat(self, flat_nav):
        returns = compute_daily_returns(flat_nav, "nav")
        sharpe = compute_sharpe(returns)
        assert sharpe == 0.0


class TestComputeSortino:
    def test_positive_for_uptrend(self, daily_nav):
        returns = compute_daily_returns(daily_nav, "nav")
        sortino = compute_sortino(returns)
        assert sortino > 0

    def test_higher_than_sharpe_when_few_down_days(self, daily_nav):
        returns = compute_daily_returns(daily_nav, "nav")
        sharpe = compute_sharpe(returns)
        sortino = compute_sortino(returns)
        assert sortino >= sharpe


class TestComputeMaxDrawdown:
    def test_returns_positive_fraction(self, daily_nav):
        dd = compute_max_drawdown(daily_nav, "nav")
        assert dd > 0
        assert dd < 1

    def test_zero_for_monotonic_increase(self):
        dates = pd.date_range("2026-01-01", periods=5, freq="D")
        df = pd.DataFrame({"nav": [100, 101, 102, 103, 104]}, index=dates)
        dd = compute_max_drawdown(df, "nav")
        assert dd == 0.0


class TestComputeTotalReturn:
    def test_correct_total_return(self, daily_nav):
        ret = compute_total_return(daily_nav, "nav")
        expected = (1000 - 980) / 980
        assert abs(ret - expected) < 1e-10


class TestComputeWinRate:
    def test_win_rate_between_0_and_1(self, daily_nav):
        returns = compute_daily_returns(daily_nav, "nav")
        wr = compute_win_rate(returns)
        assert 0 <= wr <= 1

    def test_correct_count(self, daily_nav):
        # navs: 980,982,981,984,986,985,988,990,989,992,994,993,996,998,1000
        # changes: +,-,+,+,-,+,+,-,+,+,-,+,+,+
        # 10 positive out of 14
        returns = compute_daily_returns(daily_nav, "nav")
        wr = compute_win_rate(returns)
        assert abs(wr - 10 / 14) < 1e-10


class TestComputeAllMetrics:
    def test_returns_dict_with_all_keys(self, daily_nav):
        metrics = compute_all_metrics(daily_nav, "nav")
        for key in ["total_return", "sharpe", "sortino", "max_drawdown",
                     "win_rate", "calmar", "trading_days", "annualized_return"]:
            assert key in metrics, f"Missing key: {key}"
```

**Step 2: Run tests to verify they fail**

Run: `venv/bin/pytest tests/test_metrics.py -v`
Expected: ImportError.

**Step 3: Write metrics.py**

```python
# metrics.py
"""Compute trading performance metrics from NAV/equity time series."""

import numpy as np
import pandas as pd


def compute_daily_returns(df: pd.DataFrame, value_col: str) -> pd.Series:
    """Compute daily percentage returns from a value column."""
    return df[value_col].pct_change().dropna()


def compute_sharpe(returns: pd.Series, annual_factor: float = 252) -> float:
    """Annualized Sharpe ratio (assuming risk-free rate = 0)."""
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(annual_factor))


def compute_sortino(returns: pd.Series, annual_factor: float = 252) -> float:
    """Annualized Sortino ratio (downside deviation only)."""
    if len(returns) < 2:
        return 0.0
    downside = returns[returns < 0]
    if len(downside) == 0 or downside.std() == 0:
        return float("inf") if returns.mean() > 0 else 0.0
    return float(returns.mean() / downside.std() * np.sqrt(annual_factor))


def compute_max_drawdown(df: pd.DataFrame, value_col: str) -> float:
    """Maximum peak-to-trough drawdown as a positive fraction (0 to 1)."""
    values = df[value_col]
    if len(values) < 2:
        return 0.0
    peak = values.expanding().max()
    drawdown = (peak - values) / peak
    return float(drawdown.max())


def compute_total_return(df: pd.DataFrame, value_col: str) -> float:
    """Total return as a fraction."""
    values = df[value_col]
    if len(values) < 2:
        return 0.0
    return float((values.iloc[-1] - values.iloc[0]) / values.iloc[0])


def compute_win_rate(returns: pd.Series) -> float:
    """Fraction of positive-return days."""
    if len(returns) == 0:
        return 0.0
    return float((returns > 0).sum() / len(returns))


def compute_calmar(df: pd.DataFrame, value_col: str,
                   returns: pd.Series, annual_factor: float = 252) -> float:
    """Calmar ratio: annualized return / max drawdown."""
    dd = compute_max_drawdown(df, value_col)
    if dd == 0:
        return 0.0
    ann_return = returns.mean() * annual_factor
    return float(ann_return / dd)


def compute_all_metrics(df: pd.DataFrame, value_col: str) -> dict:
    """Compute all performance metrics. Returns a dict."""
    returns = compute_daily_returns(df, value_col)
    trading_days = len(returns)
    ann_return = returns.mean() * 252 if trading_days > 0 else 0.0

    return {
        "total_return": compute_total_return(df, value_col),
        "annualized_return": float(ann_return),
        "sharpe": compute_sharpe(returns),
        "sortino": compute_sortino(returns),
        "max_drawdown": compute_max_drawdown(df, value_col),
        "win_rate": compute_win_rate(returns),
        "calmar": compute_calmar(df, value_col, returns),
        "trading_days": trading_days,
    }
```

**Step 4: Run tests to verify they pass**

Run: `venv/bin/pytest tests/test_metrics.py -v`
Expected: All pass.

**Step 5: Commit**

```bash
git add metrics.py tests/test_metrics.py
git commit -m "feat: metrics calculator (sharpe, sortino, drawdown, win rate, calmar)"
```

---

### Task 5: Chart Factory Module

**Files:**
- Create: `charts.py`

No TDD for chart factories - these are visual outputs verified by running the app.

**Step 1: Write charts.py**

```python
# charts.py
"""Plotly chart factories for the trade performance dashboard."""

import plotly.graph_objects as go
import pandas as pd

COLORS = {
    "primary": "#4A90D9",
    "green": "#00C853",
    "red": "#FF1744",
    "orange": "#FF9100",
    "purple": "#AA00FF",
    "gray": "#616161",
    "bg": "#0E1117",
    "card_bg": "#1E2130",
}

LAYOUT_DEFAULTS = dict(
    template="plotly_dark",
    paper_bgcolor=COLORS["bg"],
    plot_bgcolor=COLORS["bg"],
    font=dict(family="Inter, system-ui, sans-serif"),
    margin=dict(l=60, r=20, t=40, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)


def equity_curve(df: pd.DataFrame, value_col: str, title: str = "Equity Curve") -> go.Figure:
    """Line chart of NAV/equity over time with peak line."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index, y=df[value_col],
        mode="lines", name="NAV",
        line=dict(color=COLORS["primary"], width=2),
    ))

    peak = df[value_col].expanding().max()
    fig.add_trace(go.Scatter(
        x=df.index, y=peak,
        mode="lines", name="Peak",
        line=dict(color=COLORS["gray"], width=1, dash="dot"),
    ))

    fig.update_layout(**LAYOUT_DEFAULTS, title=title,
                      xaxis_title="Date", yaxis_title="Value ($)")
    return fig


def drawdown_chart(df: pd.DataFrame, value_col: str) -> go.Figure:
    """Area chart of drawdown percentage over time."""
    peak = df[value_col].expanding().max()
    dd = (df[value_col] - peak) / peak * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=dd,
        fill="tozeroy", name="Drawdown",
        line=dict(color=COLORS["red"], width=1),
        fillcolor="rgba(255, 23, 68, 0.3)",
    ))
    fig.update_layout(**LAYOUT_DEFAULTS, title="Drawdown",
                      xaxis_title="Date", yaxis_title="Drawdown (%)")
    return fig


def daily_returns_bar(returns: pd.Series) -> go.Figure:
    """Bar chart of daily returns colored green/red."""
    colors = [COLORS["green"] if r >= 0 else COLORS["red"] for r in returns]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=returns.index, y=returns * 100,
        marker_color=colors, name="Daily Return",
    ))
    fig.update_layout(**LAYOUT_DEFAULTS, title="Daily Returns",
                      xaxis_title="Date", yaxis_title="Return (%)")
    return fig


def positions_bar(positions: pd.DataFrame, name_col: str, value_col: str,
                  title: str = "Position P&L") -> go.Figure:
    """Horizontal bar chart of per-position P&L."""
    df = positions.sort_values(value_col)
    colors = [COLORS["green"] if v >= 0 else COLORS["red"] for v in df[value_col]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df[name_col], x=df[value_col],
        orientation="h", marker_color=colors,
    ))
    fig.update_layout(**LAYOUT_DEFAULTS, title=title,
                      xaxis_title="P&L ($)", yaxis_title="",
                      height=max(400, len(df) * 25))
    return fig


def exposure_over_time(df: pd.DataFrame) -> go.Figure:
    """Stacked area of long vs short exposure over time."""
    fig = go.Figure()

    if "long_market_value" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["long_market_value"],
            fill="tozeroy", name="Long Exposure",
            line=dict(color=COLORS["green"], width=1),
            fillcolor="rgba(0, 200, 83, 0.3)",
        ))
    if "short_market_value" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["short_market_value"].abs(),
            fill="tozeroy", name="Short Exposure",
            line=dict(color=COLORS["red"], width=1),
            fillcolor="rgba(255, 23, 68, 0.3)",
        ))
    if "margin_used" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["margin_used"],
            fill="tozeroy", name="Margin Used",
            line=dict(color=COLORS["orange"], width=1),
            fillcolor="rgba(255, 145, 0, 0.3)",
        ))

    fig.update_layout(**LAYOUT_DEFAULTS, title="Exposure Over Time",
                      xaxis_title="Date", yaxis_title="Value ($)")
    return fig


def rolling_sharpe(returns: pd.Series, window: int = 10) -> go.Figure:
    """Rolling Sharpe ratio line chart."""
    rolling_mean = returns.rolling(window).mean()
    rolling_std = returns.rolling(window).std()
    rolling = (rolling_mean / rolling_std * (252 ** 0.5)).dropna()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rolling.index, y=rolling,
        mode="lines", name=f"{window}-Day Rolling Sharpe",
        line=dict(color=COLORS["purple"], width=2),
    ))
    fig.add_hline(y=0, line_dash="dot", line_color=COLORS["gray"])
    fig.update_layout(**LAYOUT_DEFAULTS, title=f"Rolling Sharpe ({window}-Day)",
                      xaxis_title="Date", yaxis_title="Sharpe Ratio")
    return fig
```

**Step 2: Commit**

```bash
git add charts.py
git commit -m "feat: plotly chart factories (equity, drawdown, returns, positions, exposure)"
```

---

### Task 6: Main Dashboard App

**Files:**
- Create: `app.py`

**Step 1: Write app.py**

```python
# app.py
"""Trade Performance Dashboard - Streamlit app."""

import streamlit as st
import pandas as pd
from pathlib import Path

from data_loader import (
    load_oanda_nav, load_alpaca_equity, resample_daily,
    extract_oanda_positions, extract_alpaca_positions,
)
from metrics import compute_all_metrics, compute_daily_returns
from charts import (
    equity_curve, drawdown_chart, daily_returns_bar,
    positions_bar, exposure_over_time, rolling_sharpe,
)

# --- Config ---
DATA_DIR = Path("/home/kingjames/rl-trader/forex-rl/logs")
OANDA_FILE = DATA_DIR / "oanda_nav_positions.jsonl"
ALPACA_FILE = DATA_DIR / "alpaca_equity_positions.jsonl"

PHASE2_START = pd.Timestamp("2026-01-19", tz="UTC")

st.set_page_config(
    page_title="Trading Performance Dashboard",
    page_icon="chart_with_upwards_trend",
    layout="wide",
)


@st.cache_data(ttl=300)
def load_oanda():
    return load_oanda_nav(OANDA_FILE)


@st.cache_data(ttl=300)
def load_alpaca():
    return load_alpaca_equity(ALPACA_FILE)


def main():
    st.title("Algorithmic Trading Performance")
    st.caption("Live system metrics | Updated every 5 minutes")

    oanda_raw = load_oanda()
    alpaca_raw = load_alpaca()

    # --- Sidebar ---
    with st.sidebar:
        st.header("Filters")

        system = st.radio("Trading System", ["OANDA Forex", "Alpaca Equities"])

        if system == "OANDA Forex":
            raw_df = oanda_raw.copy()
            value_col = "nav"
            phase2 = st.checkbox("Phase 2 only (Jan 19+)", value=True)
            if phase2 and not raw_df.empty:
                raw_df = raw_df[raw_df.index >= PHASE2_START]
        else:
            raw_df = alpaca_raw.copy()
            value_col = "equity"
            phase2 = False

        if raw_df.empty:
            st.warning("No data available.")
            return

        min_date = raw_df.index.min().date()
        max_date = raw_df.index.max().date()
        date_range = st.date_input("Date Range",
                                   value=(min_date, max_date),
                                   min_value=min_date, max_value=max_date)
        if len(date_range) == 2:
            start, end = date_range
            raw_df = raw_df[
                (raw_df.index.date >= start) & (raw_df.index.date <= end)
            ]

        st.divider()
        st.markdown("**System Details**")
        if system == "OANDA Forex":
            st.markdown(
                "- 20 forex instruments\n"
                "- TCN + Actor-Critic RL\n"
                "- Continuous position sizing\n"
                "- 5-min decision intervals"
            )
        else:
            st.markdown(
                "- 100 long/short positions\n"
                "- US equities universe\n"
                "- Paper trading\n"
                "- 5-min rebalancing"
            )

    if raw_df.empty:
        st.warning("No data in selected range.")
        return

    daily = resample_daily(raw_df, value_col)

    if len(daily) < 2:
        st.warning("Need at least 2 days of data for metrics.")
        return

    metrics = compute_all_metrics(daily, value_col)
    returns = compute_daily_returns(daily, value_col)

    # --- Tabs ---
    tab_overview, tab_positions, tab_risk = st.tabs(
        ["Overview", "Positions", "Risk"]
    )

    # === OVERVIEW ===
    with tab_overview:
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Total Return", f"{metrics['total_return']:.2%}")
        c2.metric("Sharpe Ratio", f"{metrics['sharpe']:.2f}")
        c3.metric("Sortino Ratio", f"{metrics['sortino']:.2f}")
        c4.metric("Max Drawdown", f"{metrics['max_drawdown']:.2%}")
        c5.metric("Win Rate", f"{metrics['win_rate']:.1%}")
        c6.metric("Trading Days", f"{metrics['trading_days']}")

        label = "OANDA NAV" if system == "OANDA Forex" else "Alpaca Equity"
        st.plotly_chart(equity_curve(daily, value_col, title=f"{label} Curve"),
                        use_container_width=True)

        st.plotly_chart(daily_returns_bar(returns), use_container_width=True)

    # === POSITIONS ===
    with tab_positions:
        if system == "OANDA Forex":
            positions = extract_oanda_positions(OANDA_FILE)
            if not positions.empty:
                st.subheader("Current Positions")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Open Positions", len(positions))
                with col2:
                    total_pl = positions["pl"].sum()
                    st.metric("Total Realized P&L", f"${total_pl:+.2f}")

                st.plotly_chart(
                    positions_bar(positions, "instrument", "pl",
                                  title="Realized P&L by Instrument"),
                    use_container_width=True)
                st.plotly_chart(
                    positions_bar(positions, "instrument", "unrealized_pl",
                                  title="Unrealized P&L by Instrument"),
                    use_container_width=True)

                with st.expander("Raw Position Data"):
                    st.dataframe(positions.sort_values("pl", ascending=False),
                                 use_container_width=True)
            else:
                st.info("No position data available.")

        else:
            positions = extract_alpaca_positions(ALPACA_FILE)
            if not positions.empty:
                st.subheader("Current Positions")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Positions", len(positions))
                with col2:
                    st.metric("Long", len(positions[positions["side"] == "long"]))
                with col3:
                    st.metric("Short", len(positions[positions["side"] == "short"]))

                st.plotly_chart(
                    positions_bar(positions, "symbol", "unrealized_pl",
                                  title="Unrealized P&L by Symbol"),
                    use_container_width=True)

                with st.expander("Raw Position Data"):
                    st.dataframe(
                        positions.sort_values("unrealized_pl", ascending=False),
                        use_container_width=True)
            else:
                st.info("No position data available.")

    # === RISK ===
    with tab_risk:
        st.plotly_chart(drawdown_chart(daily, value_col),
                        use_container_width=True)

        if len(returns) >= 10:
            st.plotly_chart(
                rolling_sharpe(returns, window=min(10, len(returns) - 1)),
                use_container_width=True)

        st.plotly_chart(exposure_over_time(raw_df), use_container_width=True)

        st.subheader("Risk Summary")
        rc1, rc2, rc3, rc4 = st.columns(4)
        rc1.metric("Calmar Ratio", f"{metrics['calmar']:.2f}")
        rc2.metric("Annualized Return", f"{metrics['annualized_return']:.2%}")
        rc3.metric("Avg Daily Return",
                    f"{returns.mean():.4%}" if len(returns) > 0 else "N/A")
        rc4.metric("Daily Volatility",
                    f"{returns.std():.4%}" if len(returns) > 0 else "N/A")


if __name__ == "__main__":
    main()
```

**Step 2: Test run the app**

```bash
cd /home/kingjames/trade-performance-dashboard
venv/bin/streamlit run app.py --server.port 8501
```

Verify all three tabs render correctly in browser.

**Step 3: Commit**

```bash
git add app.py
git commit -m "feat: main dashboard app with overview, positions, and risk tabs"
```

---

### Task 7: Verification

**Step 1: Run all tests**

```bash
cd /home/kingjames/trade-performance-dashboard
venv/bin/pytest tests/ -v
```

Expected: All pass.

**Step 2: Launch app and verify against known metrics**

```bash
venv/bin/streamlit run app.py --server.port 8501
```

Verify:
- OANDA Phase 2 shows ~2.5% return, ~2.8 Sharpe, ~3.2 Sortino, ~2.8% max DD
- Equity curve shows the NAV spike above $1000 then dip then recovery
- Alpaca shows positive equity trend
- Positions tab shows per-instrument breakdown
- Risk tab shows drawdown chart and rolling Sharpe
- Dark theme applied

**Step 3: Final commit if any fixes needed**

```bash
git add -A
git commit -m "fix: any verification fixes"
```

---

## Summary

| Task | Component | Tests | Commit |
|------|-----------|-------|--------|
| 1 | Project scaffolding | - | chore: project scaffolding |
| 2 | Data loader (NAV/equity) | 9 | feat: data loader |
| 3 | Data loader (positions) | 4 | feat: position extraction |
| 4 | Metrics calculator | 12 | feat: metrics calculator |
| 5 | Chart factories | - | feat: chart factories |
| 6 | Main app | - | feat: main dashboard app |
| 7 | Verification | all | fix: if needed |

**Total: 7 tasks, ~25 tests, 6-7 commits**
