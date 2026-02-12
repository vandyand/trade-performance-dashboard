# charts.py
"""Plotly chart factories for the trade performance dashboard."""

import plotly.graph_objects as go
import plotly.express as px
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
    if "positions_value_usd" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["positions_value_usd"],
            fill="tozeroy", name="Token Positions",
            line=dict(color=COLORS["green"], width=1),
            fillcolor="rgba(0, 200, 83, 0.3)",
        ))
    if "sol_balance_usd" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["sol_balance_usd"],
            fill="tozeroy", name="SOL Balance",
            line=dict(color=COLORS["primary"], width=1),
            fillcolor="rgba(74, 144, 217, 0.3)",
        ))

    fig.update_layout(**LAYOUT_DEFAULTS, title="Exposure Over Time",
                      xaxis_title="Date", yaxis_title="Value ($)")
    return fig


def instrument_sharpe_bar(metrics_df: pd.DataFrame,
                          title: str = "Per-Instrument Sharpe Ratio") -> go.Figure:
    """Horizontal bar chart of per-instrument Sharpe ratio."""
    df = metrics_df.sort_values("sharpe")
    colors = [COLORS["green"] if s >= 0 else COLORS["red"] for s in df["sharpe"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df["instrument"], x=df["sharpe"],
        orientation="h", marker_color=colors,
        hovertemplate="<b>%{y}</b><br>Sharpe: %{x:.2f}<extra></extra>",
    ))
    fig.update_layout(**LAYOUT_DEFAULTS, title=title,
                      xaxis_title="Sharpe Ratio (annualized)", yaxis_title="",
                      height=max(400, len(df) * 28))
    return fig


def cumulative_instrument_pnl(daily_pnl: pd.DataFrame,
                              title: str = "Cumulative P&L by Instrument") -> go.Figure:
    """Interactive line chart of cumulative P&L per instrument.

    Each instrument is a separate trace with its own color.
    Click legend items to toggle, hover for values.
    """
    cum = daily_pnl.cumsum()

    # Sort columns by final cumulative P&L so best performers are listed first
    col_order = cum.iloc[-1].sort_values(ascending=False).index.tolist()

    # Use a rich color palette that cycles if needed
    palette = (px.colors.qualitative.Set2 + px.colors.qualitative.Set1
               + px.colors.qualitative.Pastel1 + px.colors.qualitative.Dark2
               + px.colors.qualitative.Set3)

    fig = go.Figure()
    for i, col in enumerate(col_order):
        color = palette[i % len(palette)]
        fig.add_trace(go.Scatter(
            x=cum.index, y=cum[col],
            mode="lines+markers",
            name=col,
            line=dict(color=color, width=2),
            marker=dict(size=5),
            hovertemplate=f"<b>{col}</b><br>"
                          "Date: %{x}<br>"
                          "Cumulative P&L: $%{y:+.2f}<extra></extra>",
        ))

    fig.add_hline(y=0, line_dash="dot", line_color=COLORS["gray"], opacity=0.5)

    layout_overrides = {k: v for k, v in LAYOUT_DEFAULTS.items() if k != "legend"}
    fig.update_layout(
        **layout_overrides,
        title=title,
        xaxis_title="Date",
        yaxis_title="Cumulative P&L ($)",
        height=500,
        legend=dict(
            orientation="v", yanchor="top", y=1, xanchor="left", x=1.02,
            font=dict(size=10),
        ),
        hovermode="x unified",
    )
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
