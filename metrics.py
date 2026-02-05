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
