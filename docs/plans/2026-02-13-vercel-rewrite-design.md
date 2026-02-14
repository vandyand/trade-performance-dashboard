# Trading Dashboard Vercel Rewrite — Design

**Date:** 2026-02-13
**Status:** Approved
**Goal:** Rewrite the Streamlit trading dashboard as a Next.js app on Vercel for instant page transitions and global edge delivery.

## Scope

1:1 port of the existing Streamlit dashboard. Same 3 trading systems (OANDA Forex, Alpaca Equities, Solana Altmemecoins), same 3 tabs (Overview, Positions, Risk), same metrics and charts. No new features.

## Data Pipeline

```
Home Server (systemd timer, every 5 min):
  JSONL files → cache_manager.py → Parquet cache (unchanged)
                                  → push_to_vercel.py (NEW)
                                      Reads Parquet, pre-computes all
                                      metrics and chart-ready data,
                                      pushes JSON to Vercel Blob
```

All computation happens server-side. The frontend does zero math.

### JSON Blobs (per system, per timeframe)

Six JSON files per system, in daily and 5-min variants (e.g., `alpaca/overview-daily.json`):

- `overview-{freq}.json` — headline metrics, equity curve series, daily returns series, system info
- `positions.json` — current positions table, instrument metrics, cumulative P&L series, top/bottom symbols (shared across timeframes since positions are point-in-time)
- `risk-{freq}.json` — drawdown series, rolling sharpe series, exposure series, risk summary metrics

### Data Shapes

```typescript
// overview-{freq}.json
{
  metrics: {
    total_return: number, sharpe: number, sortino: number,
    max_drawdown: number, win_rate: number, trading_days: number,
    annualized_return: number, calmar: number
  },
  equity_curve: { timestamps: string[], values: number[] },
  daily_returns: { timestamps: string[], values: number[] },
  system_info: { name: string, description: string, start_date: string, start_nav: number | null }
}

// positions.json
{
  positions: Array<{ symbol: string, qty: number, side: string, unrealized_pl: number, ... }>,
  instrument_metrics: {
    daily: Array<{ instrument: string, total_pnl: number, sharpe: number, ... }>,
    five_min: Array<{ instrument: string, total_pnl: number, sharpe: number, ... }>
  },
  cumulative_pnl: {
    daily: { timestamps: string[], series: Record<string, number[]> },
    five_min: { timestamps: string[], series: Record<string, number[]> }
  },
  top_bottom_symbols: string[]
}

// risk-{freq}.json
{
  drawdown: { timestamps: string[], values: number[] },
  rolling_sharpe: { timestamps: string[], values: number[] },
  exposure: { timestamps: string[], long: number[], short: number[], margin?: number[] },
  risk_metrics: { calmar: number, annualized_return: number, avg_return: number, volatility: number }
}
```

## Frontend Architecture

### Tech Stack

| Layer | Choice | Rationale |
|-------|--------|-----------|
| Framework | Next.js 14 (App Router) | Static export, file routing, Vercel-native |
| Styling | Tailwind CSS | Dark theme, no runtime CSS |
| Time-series charts | lightweight-charts v4 | TradingView library, canvas rendering, small bundle |
| Bar charts | Custom Tailwind divs | Simple horizontal bars, no library needed |
| Data fetching | SWR | Client-side cache, 60s auto-revalidation |
| Storage | Vercel Blob | Edge reads, simple PUT API from server |

### Route Structure

```
app/
├── layout.tsx              — dark theme shell, system sidebar
├── page.tsx                — redirect to /oanda
├── [system]/
│   ├── page.tsx            — Overview tab (default)
│   ├── positions/page.tsx  — Positions tab
│   └── risk/page.tsx       — Risk tab
```

System switching = URL navigation (instant).
Tab switching = client-side route change (instant).
Timeframe toggle = fetch different JSON blob, cached by SWR.

### Component Tree

```
components/
├── charts/
│   ├── EquityCurve.tsx         — Lightweight Charts line + peak overlay
│   ├── DrawdownChart.tsx       — Lightweight Charts area (red fill)
│   ├── DailyReturns.tsx        — Lightweight Charts histogram (green/red)
│   ├── RollingSharpe.tsx       — Lightweight Charts line
│   ├── CumulativePnL.tsx       — Lightweight Charts multi-line (top/bottom 10)
│   ├── ExposureChart.tsx       — Lightweight Charts stacked area
│   └── HorizontalBar.tsx       — Tailwind CSS horizontal bars (P&L, Sharpe)
├── MetricCard.tsx              — Single metric display
├── SystemSidebar.tsx           — System nav + details
└── TimeframeToggle.tsx         — Daily / 5-Min toggle
```

### Data Fetching

```typescript
// lib/data.ts — SWR hooks
function useOverview(system: string, timeframe: string) {
  return useSWR(`${BLOB_BASE}/${system}/overview-${timeframe}.json`, fetcher, {
    revalidateOnFocus: false,
    refreshInterval: 60_000,
  });
}
```

Each tab fetches only its own data. Switching tabs triggers a fetch for the new tab's data, cached by SWR.

## Push Script Design

`push_to_vercel.py` runs after `cache_manager.py` in the systemd timer:

1. Read Parquet cache files for each system
2. Compute all metrics (reusing existing `metrics.py` functions)
3. Build chart-ready JSON (timestamps + values arrays)
4. PUT to Vercel Blob via REST API (`BLOB_READ_WRITE_TOKEN` env var)

Uses the existing Parquet cache as input, so it's fast (~100ms per system).

## UX Improvements Over Streamlit

- System switching: instant URL navigation vs full script rerun
- Tab switching: client-side route vs Streamlit rerun
- Timeframe toggle: SWR-cached fetch vs Streamlit rerun
- No Cloudflare tunnel: Vercel edge serves globally with sub-100ms latency
- Charts render on canvas (Lightweight Charts) vs JSON-over-websocket (Plotly)
- Dark theme preserved via Tailwind
