# tests/test_cache_manager.py
import json
import pytest
import pandas as pd
from pathlib import Path

from cache_manager import (
    _read_jsonl_from_offset,
    _extract_oanda_data,
    _extract_alpaca_data,
    _extract_solana_data,
    _snapshots_to_cumulative_df,
    _nav_rows_to_df,
    refresh_cache,
    refresh_all,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def oanda_records():
    """Two raw OANDA snapshot dicts."""
    return [
        {
            "event": "oanda_nav_positions_snapshot",
            "time_iso": "2026-01-19T10:00:00+00:00",
            "account": {
                "NAV": "980.00", "balance": "978.00",
                "pl": "-22.00", "unrealizedPL": "2.00",
                "financing": "-5.00", "openTradeCount": 5,
                "marginUsed": "50.00", "positionValue": "1200.00",
            },
            "positions": [
                {
                    "instrument": "EUR_USD",
                    "long": {"units": "100", "pl": "5.00", "unrealizedPL": "1.00"},
                    "short": {"units": "0", "pl": "-2.00", "unrealizedPL": "0.00"},
                    "pl": "3.00", "unrealizedPL": "1.00",
                },
            ],
        },
        {
            "event": "oanda_nav_positions_snapshot",
            "time_iso": "2026-01-19T10:05:00+00:00",
            "account": {
                "NAV": "981.50", "balance": "978.00",
                "pl": "-22.00", "unrealizedPL": "3.50",
                "financing": "-5.00", "openTradeCount": 5,
                "marginUsed": "51.00", "positionValue": "1210.00",
            },
            "positions": [
                {
                    "instrument": "EUR_USD",
                    "long": {"units": "100", "pl": "5.50", "unrealizedPL": "1.50"},
                    "short": {"units": "0", "pl": "-2.00", "unrealizedPL": "0.00"},
                    "pl": "3.50", "unrealizedPL": "1.50",
                },
            ],
        },
    ]


@pytest.fixture
def oanda_jsonl(tmp_path, oanda_records):
    f = tmp_path / "oanda.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in oanda_records) + "\n")
    return f


@pytest.fixture
def alpaca_records():
    return [
        {
            "event": "alpaca_equity_snapshot",
            "time_iso": "2026-02-03T15:00:00+00:00",
            "account": {
                "equity": 100000.0, "cash": 80000.0,
                "long_market_value": 15000.0, "short_market_value": -5000.0,
            },
            "positions": [
                {
                    "symbol": "AAPL", "qty": 10.0, "side": "PositionSide.LONG",
                    "market_value": 2000.0, "cost_basis": 1900.0,
                    "avg_entry_price": 190.0,
                    "unrealized_pl": 100.0, "unrealized_plpc": 0.0526,
                },
            ],
            "totals": {"positions_count": 1},
        },
        {
            "event": "alpaca_equity_snapshot",
            "time_iso": "2026-02-03T15:05:00+00:00",
            "account": {
                "equity": 100050.0, "cash": 80000.0,
                "long_market_value": 15050.0, "short_market_value": -5000.0,
            },
            "positions": [
                {
                    "symbol": "AAPL", "qty": 10.0, "side": "PositionSide.LONG",
                    "market_value": 2050.0, "cost_basis": 1900.0,
                    "avg_entry_price": 190.0,
                    "unrealized_pl": 150.0, "unrealized_plpc": 0.0789,
                },
            ],
            "totals": {"positions_count": 1},
        },
    ]


@pytest.fixture
def alpaca_jsonl(tmp_path, alpaca_records):
    f = tmp_path / "alpaca.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in alpaca_records) + "\n")
    return f


@pytest.fixture
def solana_records():
    return [
        {
            "event": "solana_account_snapshot",
            "time_iso": "2026-02-09T18:00:00+00:00",
            "account": {
                "nav": 11.87, "sol_balance_usd": 10.78,
                "positions_value_usd": 1.09, "positions_count": 1,
                "total_unrealized_pnl": 0.001,
            },
            "positions": [
                {
                    "symbol": "BIRB", "mint": "G7vQ...",
                    "value_usd": 0.37, "entry_usd": 0.369,
                    "unrealized_pnl": 0.001, "price_impact_pct": 0.5,
                },
            ],
        },
        {
            "event": "solana_account_snapshot",
            "time_iso": "2026-02-09T18:05:00+00:00",
            "account": {
                "nav": 11.90, "sol_balance_usd": 10.80,
                "positions_value_usd": 1.10, "positions_count": 1,
                "total_unrealized_pnl": 0.002,
            },
            "positions": [
                {
                    "symbol": "BIRB", "mint": "G7vQ...",
                    "value_usd": 0.38, "entry_usd": 0.369,
                    "unrealized_pnl": 0.002, "price_impact_pct": 0.5,
                },
            ],
        },
    ]


@pytest.fixture
def solana_jsonl(tmp_path, solana_records):
    f = tmp_path / "solana.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in solana_records) + "\n")
    return f


# ---------------------------------------------------------------------------
# _read_jsonl_from_offset
# ---------------------------------------------------------------------------

class TestReadJsonlFromOffset:
    def test_reads_all_from_start(self, oanda_jsonl):
        records, offset = _read_jsonl_from_offset(oanda_jsonl, 0)
        assert len(records) == 2
        assert offset > 0

    def test_reads_from_byte_offset(self, oanda_jsonl):
        # Read first pass to get offset after first record
        all_records, full_offset = _read_jsonl_from_offset(oanda_jsonl, 0)
        assert len(all_records) == 2

        # Find offset after first line
        with open(oanda_jsonl, "rb") as f:
            first_line = f.readline()
            mid_offset = f.tell()

        records, new_offset = _read_jsonl_from_offset(oanda_jsonl, mid_offset)
        assert len(records) == 1
        assert new_offset == full_offset

    def test_returns_empty_for_missing_file(self, tmp_path):
        records, offset = _read_jsonl_from_offset(tmp_path / "nope.jsonl")
        assert records == []
        assert offset == 0

    def test_skips_malformed_lines(self, tmp_path):
        f = tmp_path / "bad.jsonl"
        f.write_text('{"good": true}\nthis is not json\n{"also_good": true}\n')
        records, _ = _read_jsonl_from_offset(f)
        assert len(records) == 2

    def test_handles_truncated_file(self, oanda_jsonl):
        # If offset > file size, should reset and read all
        records, offset = _read_jsonl_from_offset(oanda_jsonl, 999999999)
        assert len(records) == 2


# ---------------------------------------------------------------------------
# Extractors
# ---------------------------------------------------------------------------

class TestExtractOandaData:
    def test_extracts_nav_rows(self, oanda_records):
        nav, _, _ = _extract_oanda_data(oanda_records)
        assert len(nav) == 2
        assert nav[0]["nav"] == 980.00
        assert nav[1]["nav"] == 981.50

    def test_extracts_instrument_snapshots(self, oanda_records):
        _, snapshots, _ = _extract_oanda_data(oanda_records)
        assert len(snapshots) == 2
        ts, pnl = snapshots[0]
        assert "EUR_USD" in pnl
        # cumulative = pl + unrealizedPL = 3.00 + 1.00
        assert pnl["EUR_USD"] == 4.0

    def test_extracts_latest_positions(self, oanda_records):
        _, _, positions = _extract_oanda_data(oanda_records)
        assert len(positions) == 1
        assert positions[0]["instrument"] == "EUR_USD"
        # From second (latest) record
        assert positions[0]["pl"] == 3.50

    def test_handles_empty_records(self):
        nav, snapshots, positions = _extract_oanda_data([])
        assert nav == []
        assert snapshots == []
        assert positions == []


class TestExtractAlpacaData:
    def test_extracts_equity_rows(self, alpaca_records):
        nav, _, _ = _extract_alpaca_data(alpaca_records)
        assert len(nav) == 2
        assert nav[0]["equity"] == 100000.0

    def test_extracts_instrument_snapshots(self, alpaca_records):
        _, snapshots, _ = _extract_alpaca_data(alpaca_records)
        assert len(snapshots) == 2
        _, pnl = snapshots[0]
        assert pnl["AAPL"] == 100.0

    def test_extracts_latest_positions(self, alpaca_records):
        _, _, positions = _extract_alpaca_data(alpaca_records)
        assert len(positions) == 1
        assert positions[0]["symbol"] == "AAPL"
        assert positions[0]["unrealized_pl"] == 150.0


class TestExtractSolanaData:
    def test_filters_by_event_type(self):
        records = [
            {"event": "other_event", "time_iso": "2026-01-01T00:00:00+00:00",
             "account": {"nav": 1}},
            {"event": "solana_account_snapshot",
             "time_iso": "2026-02-09T18:00:00+00:00",
             "account": {"nav": 11.87, "sol_balance_usd": 10.78,
                         "positions_value_usd": 1.09, "positions_count": 1,
                         "total_unrealized_pnl": 0.001},
             "positions": []},
        ]
        nav, _, _ = _extract_solana_data(records)
        assert len(nav) == 1

    def test_extracts_nav_rows(self, solana_records):
        nav, _, _ = _extract_solana_data(solana_records)
        assert len(nav) == 2
        assert nav[0]["nav"] == 11.87

    def test_extracts_latest_positions(self, solana_records):
        _, _, positions = _extract_solana_data(solana_records)
        assert len(positions) == 1
        assert positions[0]["symbol"] == "BIRB"


# ---------------------------------------------------------------------------
# DataFrame builders
# ---------------------------------------------------------------------------

class TestSnapshotsToCumulativeDf:
    def test_creates_wide_dataframe(self):
        snapshots = [
            (pd.Timestamp("2026-01-19T10:00:00+00:00"), {"EUR_USD": 4.0, "GBP_USD": 2.0}),
            (pd.Timestamp("2026-01-19T10:05:00+00:00"), {"EUR_USD": 5.0, "GBP_USD": 3.0}),
        ]
        df = _snapshots_to_cumulative_df(snapshots)
        assert len(df) == 2
        assert "EUR_USD" in df.columns
        assert "GBP_USD" in df.columns
        assert str(df.index.tz) == "UTC"

    def test_fills_missing_instruments(self):
        snapshots = [
            (pd.Timestamp("2026-01-19T10:00:00+00:00"), {"EUR_USD": 4.0}),
            (pd.Timestamp("2026-01-19T10:05:00+00:00"), {"EUR_USD": 5.0, "GBP_USD": 3.0}),
        ]
        df = _snapshots_to_cumulative_df(snapshots)
        assert df.loc[df.index[0], "GBP_USD"] == 0.0

    def test_empty_for_no_snapshots(self):
        df = _snapshots_to_cumulative_df([])
        assert df.empty


# ---------------------------------------------------------------------------
# refresh_cache
# ---------------------------------------------------------------------------

class TestRefreshCache:
    def test_full_rebuild_creates_all_files(self, oanda_jsonl, tmp_path):
        cache_dir = tmp_path / "cache"
        refresh_cache("oanda", full_rebuild=True,
                       cache_dir=cache_dir, jsonl_path=oanda_jsonl)

        system_dir = cache_dir / "oanda"
        assert (system_dir / "nav.parquet").exists()
        assert (system_dir / "instrument_pnl.parquet").exists()
        assert (system_dir / "latest_positions.json").exists()
        assert (system_dir / "meta.json").exists()

        nav_df = pd.read_parquet(system_dir / "nav.parquet")
        assert len(nav_df) == 2
        assert nav_df["nav"].iloc[0] == 980.00

    def test_incremental_appends_new_records(self, tmp_path):
        # Write 2 records
        records = [
            {
                "event": "oanda_nav_positions_snapshot",
                "time_iso": "2026-01-19T10:00:00+00:00",
                "account": {
                    "NAV": "980", "balance": "978", "pl": "-22",
                    "unrealizedPL": "2", "financing": "-5",
                    "openTradeCount": 5, "marginUsed": "50",
                    "positionValue": "1200",
                },
                "positions": [
                    {"instrument": "EUR_USD",
                     "long": {"units": "100", "pl": "3", "unrealizedPL": "1"},
                     "short": {"units": "0", "pl": "0", "unrealizedPL": "0"},
                     "pl": "3", "unrealizedPL": "1"},
                ],
            },
            {
                "event": "oanda_nav_positions_snapshot",
                "time_iso": "2026-01-19T10:05:00+00:00",
                "account": {
                    "NAV": "981", "balance": "978", "pl": "-22",
                    "unrealizedPL": "3", "financing": "-5",
                    "openTradeCount": 5, "marginUsed": "51",
                    "positionValue": "1210",
                },
                "positions": [
                    {"instrument": "EUR_USD",
                     "long": {"units": "100", "pl": "3.5", "unrealizedPL": "1.5"},
                     "short": {"units": "0", "pl": "0", "unrealizedPL": "0"},
                     "pl": "3.5", "unrealizedPL": "1.5"},
                ],
            },
        ]
        jsonl_path = tmp_path / "oanda.jsonl"
        jsonl_path.write_text(
            "\n".join(json.dumps(r) for r in records) + "\n")

        cache_dir = tmp_path / "cache"

        # Full build
        refresh_cache("oanda", full_rebuild=True,
                       cache_dir=cache_dir, jsonl_path=jsonl_path)
        nav1 = pd.read_parquet(cache_dir / "oanda" / "nav.parquet")
        assert len(nav1) == 2

        # Append a third record
        third = {
            "event": "oanda_nav_positions_snapshot",
            "time_iso": "2026-01-19T10:10:00+00:00",
            "account": {
                "NAV": "982", "balance": "978", "pl": "-22",
                "unrealizedPL": "4", "financing": "-5",
                "openTradeCount": 5, "marginUsed": "52",
                "positionValue": "1220",
            },
            "positions": [
                {"instrument": "EUR_USD",
                 "long": {"units": "100", "pl": "4", "unrealizedPL": "2"},
                 "short": {"units": "0", "pl": "0", "unrealizedPL": "0"},
                 "pl": "4", "unrealizedPL": "2"},
            ],
        }
        with open(jsonl_path, "a") as f:
            f.write(json.dumps(third) + "\n")

        # Incremental update
        refresh_cache("oanda", cache_dir=cache_dir, jsonl_path=jsonl_path)
        nav2 = pd.read_parquet(cache_dir / "oanda" / "nav.parquet")
        assert len(nav2) == 3
        assert nav2["nav"].iloc[-1] == 982.0

    def test_incremental_handles_new_instruments(self, tmp_path):
        # First record: only EUR_USD
        records = [
            {
                "event": "oanda_nav_positions_snapshot",
                "time_iso": "2026-01-19T10:00:00+00:00",
                "account": {
                    "NAV": "980", "balance": "978", "pl": "-22",
                    "unrealizedPL": "2", "financing": "-5",
                    "openTradeCount": 5, "marginUsed": "50",
                    "positionValue": "1200",
                },
                "positions": [
                    {"instrument": "EUR_USD",
                     "long": {"units": "100", "pl": "3", "unrealizedPL": "1"},
                     "short": {"units": "0", "pl": "0", "unrealizedPL": "0"},
                     "pl": "3", "unrealizedPL": "1"},
                ],
            },
        ]
        jsonl_path = tmp_path / "oanda.jsonl"
        jsonl_path.write_text(json.dumps(records[0]) + "\n")

        cache_dir = tmp_path / "cache"
        refresh_cache("oanda", full_rebuild=True,
                       cache_dir=cache_dir, jsonl_path=jsonl_path)

        # Append record with new instrument GBP_USD
        new_rec = {
            "event": "oanda_nav_positions_snapshot",
            "time_iso": "2026-01-19T10:05:00+00:00",
            "account": {
                "NAV": "981", "balance": "978", "pl": "-22",
                "unrealizedPL": "3", "financing": "-5",
                "openTradeCount": 5, "marginUsed": "51",
                "positionValue": "1210",
            },
            "positions": [
                {"instrument": "EUR_USD",
                 "long": {"units": "100", "pl": "4", "unrealizedPL": "1"},
                 "short": {"units": "0", "pl": "0", "unrealizedPL": "0"},
                 "pl": "4", "unrealizedPL": "1"},
                {"instrument": "GBP_USD",
                 "long": {"units": "50", "pl": "1", "unrealizedPL": "0.5"},
                 "short": {"units": "0", "pl": "0", "unrealizedPL": "0"},
                 "pl": "1", "unrealizedPL": "0.5"},
            ],
        }
        with open(jsonl_path, "a") as f:
            f.write(json.dumps(new_rec) + "\n")

        refresh_cache("oanda", cache_dir=cache_dir, jsonl_path=jsonl_path)

        pnl_df = pd.read_parquet(cache_dir / "oanda" / "instrument_pnl.parquet")
        assert "GBP_USD" in pnl_df.columns
        # First row should have 0.0 for GBP_USD (didn't exist yet)
        assert pnl_df["GBP_USD"].iloc[0] == 0.0

    def test_skips_when_no_new_data(self, oanda_jsonl, tmp_path):
        cache_dir = tmp_path / "cache"
        refresh_cache("oanda", full_rebuild=True,
                       cache_dir=cache_dir, jsonl_path=oanda_jsonl)

        # Second run — no new data
        refresh_cache("oanda", cache_dir=cache_dir, jsonl_path=oanda_jsonl)

        nav = pd.read_parquet(cache_dir / "oanda" / "nav.parquet")
        assert len(nav) == 2  # unchanged


class TestRefreshAll:
    def test_refreshes_all_systems(self, oanda_jsonl, alpaca_jsonl,
                                    solana_jsonl, tmp_path, monkeypatch):
        cache_dir = tmp_path / "cache"
        monkeypatch.setattr("cache_manager.CACHE_DIR", cache_dir)
        monkeypatch.setattr("cache_manager.SYSTEMS", {
            "oanda": {"jsonl_path": oanda_jsonl},
            "alpaca": {"jsonl_path": alpaca_jsonl},
            "solana": {"jsonl_path": solana_jsonl},
        })

        refresh_all(full_rebuild=True)

        for system in ("oanda", "alpaca", "solana"):
            assert (cache_dir / system / "nav.parquet").exists()
            assert (cache_dir / system / "meta.json").exists()
