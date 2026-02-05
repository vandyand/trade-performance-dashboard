# Trade Performance Dashboard

Live performance dashboard for two algorithmic trading systems, built with Streamlit and Plotly.

## Systems

- **OANDA Forex** — 20 currency pairs, TCN + Actor-Critic RL agent, daily position sizing
- **Alpaca Equities** — 100 long/short US equities, daily rebalancing, paper trading

## Features

- Equity curve with peak overlay
- Per-instrument cumulative P&L (interactive, toggleable)
- Per-instrument Sharpe & Sortino ratios
- Drawdown analysis
- Rolling Sharpe ratio
- Long/short exposure over time
- Daily returns breakdown
- Current positions with realized & unrealized P&L

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Data

The dashboard reads JSONL log files produced by systemd samplers that snapshot account state every 5 minutes. Data is cached and refreshed on a 5-minute TTL.

## Tests

```bash
pytest tests/ -v
```
