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
