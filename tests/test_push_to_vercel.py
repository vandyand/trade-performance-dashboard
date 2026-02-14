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
