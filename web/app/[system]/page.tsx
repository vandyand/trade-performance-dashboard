"use client";

import { useParams } from "next/navigation";
import { useOverview } from "@/lib/data";
import { useTimeframe } from "./SystemLayoutClient";
import MetricCard from "@/components/MetricCard";
import EquityCurve from "@/components/charts/EquityCurve";
import DailyReturns from "@/components/charts/DailyReturns";
import { SYSTEMS } from "@/lib/systems";
import { fmtPct, fmtNumber } from "@/lib/format";
import type { SystemSlug } from "@/lib/types";

export default function OverviewPage() {
  const { system } = useParams() as { system: SystemSlug };
  const timeframe = useTimeframe();
  const { data, isLoading, error } = useOverview(system, timeframe);
  const config = SYSTEMS[system];

  if (isLoading) {
    return <div className="text-white/50 animate-pulse">Loading overview...</div>;
  }
  if (error || !data) {
    return <div className="text-accent-red">Failed to load data</div>;
  }

  const m = data.metrics;
  const periodLabel = timeframe === "5min" ? "Periods" : "Trading Days";
  const curveLabel = `${config.valueName} (${timeframe === "daily" ? "Daily" : "5-Minute"})`;

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
        <MetricCard label="Total Return" value={fmtPct(m.total_return)} />
        <MetricCard label="Sharpe Ratio" value={fmtNumber(m.sharpe)} />
        <MetricCard label="Sortino Ratio" value={fmtNumber(m.sortino)} />
        <MetricCard label="Max Drawdown" value={fmtPct(m.max_drawdown)} />
        <MetricCard label="Win Rate" value={fmtPct(m.win_rate, 1)} />
        <MetricCard label={periodLabel} value={String(m.trading_days)} />
      </div>

      <EquityCurve key={`equity-${timeframe}`} data={data.equity_curve} title={`${curveLabel} Curve`} />
      <DailyReturns key={`returns-${timeframe}`} data={data.daily_returns} />
    </div>
  );
}
