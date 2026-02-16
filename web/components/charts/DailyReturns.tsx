"use client";

import type { TimeSeries } from "@/lib/types";
import { HistogramSeries } from "lightweight-charts";
import ChartContainer from "./ChartContainer";
import { toChartTime } from "@/lib/chartTime";

interface Props {
  data: TimeSeries;
}

export default function DailyReturns({ data }: Props) {
  return (
    <div>
      <h3 className="text-sm font-medium text-white/70 mb-2">Daily Returns</h3>
      <ChartContainer
        onChart={(chart) => {
          const series = chart.addSeries(HistogramSeries, {
            priceFormat: { type: "percent" },
          });
          series.setData(
            data.timestamps.map((t, i) => ({
              time: toChartTime(t),
              value: data.values[i] * 100,
              color: data.values[i] >= 0 ? "#00C853" : "#FF1744",
            }))
          );
          chart.timeScale().fitContent();
        }}
      />
    </div>
  );
}
