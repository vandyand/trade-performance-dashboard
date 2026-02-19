export interface TimeSeries {
  timestamps: string[];
  values: number[];
}

export interface OverviewData {
  metrics: {
    total_return: number;
    sharpe: number;
    sortino: number;
    max_drawdown: number;
    win_rate: number;
    trading_days: number;
    annualized_return: number;
    calmar: number;
  };
  equity_curve: TimeSeries;
  daily_returns: TimeSeries;
  system_info: {
    name: string;
    description: string;
    start_date: string | null;
    start_nav: number | null;
  };
}

export interface Position {
  symbol: string;
  [key: string]: string | number;
}

export interface InstrumentMetric {
  instrument: string;
  total_pnl: number;
  avg_daily_pnl: number;
  sharpe: number;
  sortino: number;
  win_rate: number;
  trading_days: number;
}

export interface PositionsData {
  positions: Position[];
  instrument_metrics: {
    daily: InstrumentMetric[];
    five_min: InstrumentMetric[];
  };
  cumulative_pnl: {
    daily: { timestamps: string[]; series: Record<string, number[]> };
    five_min: { timestamps: string[]; series: Record<string, number[]> };
  };
  top_bottom_symbols: string[];
  summary: {
    total: number;
    long: number;
    short: number;
    algo_pnl: number | null;
    positions_value: number | null;
    total_unrealized_pnl: number | null;
  };
}

export interface RiskData {
  drawdown: TimeSeries;
  rolling_sharpe: TimeSeries;
  exposure: {
    timestamps: string[];
    series: Record<string, number[]>;
  };
  risk_metrics: {
    calmar: number;
    annualized_return: number;
    avg_return: number;
    volatility: number;
  };
}

export type SystemSlug = "oanda" | "alpaca" | "solana" | "kalshi";
export type Timeframe = "daily" | "5min";
