"use client";

import type { TimeSeries } from "@/lib/types";
import { AreaSeries } from "lightweight-charts";
import ChartContainer from "./ChartContainer";
import { toChartTime } from "@/lib/chartTime";

interface Props {
  data: TimeSeries;
}

export default function DrawdownChart({ data }: Props) {
  return (
    <div>
      <h3 className="text-sm font-medium text-white/70 mb-2">Drawdown</h3>
      <ChartContainer
        onChart={(chart) => {
          const series = chart.addSeries(AreaSeries, {
            lineColor: "#FF1744",
            topColor: "rgba(255, 23, 68, 0.4)",
            bottomColor: "rgba(255, 23, 68, 0.0)",
            lineWidth: 1,
            priceFormat: { type: "percent" },
          });
          series.setData(
            data.timestamps.map((t, i) => ({
              time: toChartTime(t),
              value: data.values[i] * 100,
            }))
          );
          chart.timeScale().fitContent();
        }}
      />
    </div>
  );
}
