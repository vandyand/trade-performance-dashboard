"use client";

import type { TimeSeries } from "@/lib/types";
import { LineSeries } from "lightweight-charts";
import ChartContainer from "./ChartContainer";
import { toChartTime } from "@/lib/chartTime";

interface Props {
  data: TimeSeries;
  window: number;
}

export default function RollingSharpe({ data, window: windowSize }: Props) {
  if (data.timestamps.length === 0) return null;

  return (
    <div>
      <h3 className="text-sm font-medium text-white/70 mb-2">
        Rolling Sharpe ({windowSize}-Period)
      </h3>
      <ChartContainer
        onChart={(chart) => {
          const series = chart.addSeries(LineSeries, {
            color: "#AA00FF",
            lineWidth: 2,
          });
          series.setData(
            data.timestamps.map((t, i) => ({
              time: toChartTime(t),
              value: data.values[i],
            }))
          );
          chart.timeScale().fitContent();
        }}
      />
    </div>
  );
}
