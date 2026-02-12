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


def compute_instrument_metrics(daily_pnl: pd.DataFrame,
                               annual_factor: float = 252) -> pd.DataFrame:
    """Compute per-instrument metrics from period P&L DataFrame.

    Input: DataFrame with date/datetime index, one column per instrument.
    Returns: DataFrame with columns: instrument, total_pnl, avg_daily_pnl,
    sharpe, sortino, win_rate, trading_days.
    """
    if daily_pnl.empty:
        return pd.DataFrame()

    results = []
    for col in daily_pnl.columns:
        s = daily_pnl[col].dropna()
        n = len(s)
        if n == 0:
            continue

        total = s.sum()
        avg = s.mean()
        std = s.std() if n > 1 else 0.0
        downside = s[s < 0]
        down_std = downside.std() if len(downside) > 1 else 0.0

        if n < 2 or std == 0:
            sharpe = 0.0
        else:
            sharpe = avg / std * np.sqrt(annual_factor)

        if n < 2 or down_std == 0:
            sortino = float("inf") if avg > 0 else 0.0
        else:
            sortino = avg / down_std * np.sqrt(annual_factor)

        win_rate = (s > 0).sum() / n

        results.append({
            "instrument": col,
            "total_pnl": round(total, 2),
            "avg_daily_pnl": round(avg, 2),
            "sharpe": round(sharpe, 2),
            "sortino": round(sortino, 2),
            "win_rate": round(win_rate, 3),
            "trading_days": n,
        })

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results).sort_values("sharpe", ascending=False).reset_index(drop=True)


def compute_all_metrics(df: pd.DataFrame, value_col: str,
                        annual_factor: float = 252) -> dict:
    """Compute all performance metrics. Returns a dict."""
    returns = compute_daily_returns(df, value_col)
    trading_days = len(returns)
    ann_return = returns.mean() * annual_factor if trading_days > 0 else 0.0

    return {
        "total_return": compute_total_return(df, value_col),
        "annualized_return": float(ann_return),
        "sharpe": compute_sharpe(returns, annual_factor),
        "sortino": compute_sortino(returns, annual_factor),
        "max_drawdown": compute_max_drawdown(df, value_col),
        "win_rate": compute_win_rate(returns),
        "calmar": compute_calmar(df, value_col, returns, annual_factor),
        "trading_days": trading_days,
    }
