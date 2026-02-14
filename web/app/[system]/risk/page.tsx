"use client";

import { useParams } from "next/navigation";
import { useRisk } from "@/lib/data";
import { useTimeframe } from "../SystemLayoutClient";
import MetricCard from "@/components/MetricCard";
import DrawdownChart from "@/components/charts/DrawdownChart";
import RollingSharpe from "@/components/charts/RollingSharpe";
import ExposureChart from "@/components/charts/ExposureChart";
import { fmtPct, fmtNumber } from "@/lib/format";
import type { SystemSlug } from "@/lib/types";

export default function RiskPage() {
  const { system } = useParams() as { system: SystemSlug };
  const timeframe = useTimeframe();
  const { data, isLoading, error } = useRisk(system, timeframe);

  if (isLoading) {
    return <div className="text-white/50 animate-pulse">Loading risk data...</div>;
  }
  if (error || !data) {
    return <div className="text-accent-red">Failed to load data</div>;
  }

  const m = data.risk_metrics;
  const retLabel = timeframe === "5min" ? "Avg 5-Min Return" : "Avg Daily Return";
  const volLabel = timeframe === "5min" ? "5-Min Volatility" : "Daily Volatility";
  const rollingWindow = timeframe === "5min" ? 288 : 10;

  return (
    <div className="space-y-6">
      <DrawdownChart key={`dd-${timeframe}`} data={data.drawdown} />
      <RollingSharpe key={`sharpe-${timeframe}`} data={data.rolling_sharpe} window={rollingWindow} />
      <ExposureChart key={`exp-${timeframe}`} timestamps={data.exposure.timestamps} series={data.exposure.series} />

      <div>
        <h2 className="text-lg font-semibold mb-3">Risk Summary</h2>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <MetricCard label="Calmar Ratio" value={fmtNumber(m.calmar)} />
          <MetricCard label="Annualized Return" value={fmtPct(m.annualized_return)} />
          <MetricCard label={retLabel} value={fmtPct(m.avg_return, 4)} />
          <MetricCard label={volLabel} value={fmtPct(m.volatility, 4)} />
        </div>
      </div>
    </div>
  );
}
