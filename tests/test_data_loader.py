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
