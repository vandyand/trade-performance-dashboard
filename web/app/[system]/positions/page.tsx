"use client";

import { useParams } from "next/navigation";
import { usePositions } from "@/lib/data";
import { useTimeframe } from "../SystemLayoutClient";
import MetricCard from "@/components/MetricCard";
import HorizontalBar from "@/components/HorizontalBar";
import CumulativePnL from "@/components/charts/CumulativePnL";
import DataTable from "@/components/DataTable";
import { fmtDollar, fmtPct, fmtNumber, fmtSignedDollar } from "@/lib/format";
import type { SystemSlug, InstrumentMetric } from "@/lib/types";

export default function PositionsPage() {
  const { system } = useParams() as { system: SystemSlug };
  const timeframe = useTimeframe();
  const { data, isLoading, error } = usePositions(system);

  if (isLoading) {
    return <div className="text-white/50 animate-pulse">Loading positions...</div>;
  }
  if (error || !data) {
    return <div className="text-accent-red">Failed to load data</div>;
  }

  const freq = timeframe === "daily" ? "daily" : "five_min";
  const instMetrics: InstrumentMetric[] = data.instrument_metrics[freq];
  const cumPnl = data.cumulative_pnl[freq];
  const pnlLabel = timeframe === "5min" ? "Avg 5-Min P&L" : "Avg Daily P&L";
  const periodsLabel = timeframe === "5min" ? "Periods" : "Days";

  const summaryCards = [];
  if (system === "oanda") {
    summaryCards.push(
      { label: "Open Positions", value: String(data.summary.total) },
    );
    if (data.summary.algo_pnl != null) {
      summaryCards.push({ label: "Algorithm P&L", value: fmtSignedDollar(data.summary.algo_pnl) });
    }
  } else if (system === "alpaca") {
    summaryCards.push(
      { label: "Total Positions", value: String(data.summary.total) },
      { label: "Long", value: String(data.summary.long) },
      { label: "Short", value: String(data.summary.short) },
    );
  } else if (system === "kalshi") {
    summaryCards.push(
      { label: "Active Agents", value: String(data.summary.total) },
    );
    if (data.summary.positions_value != null) {
      summaryCards.push({ label: "Positions Value", value: fmtDollar(data.summary.positions_value) });
    }
    if (data.summary.total_unrealized_pnl != null) {
      summaryCards.push({ label: "Unrealized P&L", value: fmtSignedDollar(data.summary.total_unrealized_pnl) });
    }
  } else {
    summaryCards.push(
      { label: "Open Positions", value: String(data.summary.total) },
    );
    if (data.summary.positions_value != null) {
      summaryCards.push({ label: "Positions Value", value: fmtDollar(data.summary.positions_value) });
    }
    if (data.summary.total_unrealized_pnl != null) {
      summaryCards.push({ label: "Unrealized P&L", value: fmtSignedDollar(data.summary.total_unrealized_pnl) });
    }
  }

  const plField = (system === "solana" || system === "kalshi") ? "unrealized_pnl" : "unrealized_pl";
  const symbolField = system === "oanda" ? "instrument" : "symbol";

  const sortedPositions = [...data.positions].sort(
    (a, b) => (a[plField] as number) - (b[plField] as number)
  );

  const metricCols = [
    { key: "instrument", label: system === "oanda" ? "Instrument" : "Symbol" },
    { key: "total_pnl", label: "Total P&L", align: "right" as const, format: (v: number) => fmtDollar(v) },
    { key: "avg_daily_pnl", label: pnlLabel, align: "right" as const, format: (v: number) => fmtDollar(v) },
    { key: "sharpe", label: "Sharpe", align: "right" as const, format: (v: number) => fmtNumber(v) },
    { key: "sortino", label: "Sortino", align: "right" as const, format: (v: number) => fmtNumber(v) },
    { key: "win_rate", label: "Win Rate", align: "right" as const, format: (v: number) => fmtPct(v, 1) },
    { key: "trading_days", label: periodsLabel, align: "right" as const },
  ];

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-lg font-semibold mb-3">Current Positions</h2>
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
          {summaryCards.map((c) => (
            <MetricCard key={c.label} label={c.label} value={c.value} />
          ))}
        </div>
      </div>

      {cumPnl.timestamps.length > 0 && (
        <CumulativePnL
          key={`pnl-${timeframe}`}
          timestamps={cumPnl.timestamps}
          series={cumPnl.series}
          title={`Cumulative P&L — ${system === "alpaca" ? "Top/Bottom 10" : "By Instrument"} (${timeframe === "daily" ? "Daily" : "5-Minute"})`}
        />
      )}

      <HorizontalBar
        items={sortedPositions.map((p) => ({
          label: p[symbolField] as string,
          value: p[plField] as number,
        }))}
        title={system === "oanda" ? "Unrealized P&L by Instrument" : "Unrealized P&L by Symbol"}
      />

      <div>
        <h2 className="text-lg font-semibold mb-3">Per-Instrument Performance</h2>
        <HorizontalBar
          items={[...instMetrics]
            .sort((a, b) => a.sharpe - b.sharpe)
            .map((m) => ({ label: m.instrument, value: m.sharpe }))}
          title={`Per-Instrument Sharpe (${timeframe === "daily" ? "Daily" : "5-Minute"})`}
          formatValue={(v) => v.toFixed(2)}
        />
        <div className="mt-6">
          <DataTable columns={metricCols} rows={instMetrics} />
        </div>
      </div>
    </div>
  );
}
