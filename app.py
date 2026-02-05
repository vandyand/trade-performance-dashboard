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
