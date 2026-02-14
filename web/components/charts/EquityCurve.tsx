"use client";

import type { TimeSeries } from "@/lib/types";
import { LineSeries } from "lightweight-charts";
import ChartContainer from "./ChartContainer";

interface Props {
  data: TimeSeries;
  title: string;
}

export default function EquityCurve({ data, title }: Props) {
  return (
    <div>
      <h3 className="text-sm font-medium text-white/70 mb-2">{title}</h3>
      <ChartContainer
        onChart={(chart) => {
          const series = chart.addSeries(LineSeries, {
            color: "#4A90D9",
            lineWidth: 2,
            priceFormat: { type: "price", precision: 2, minMove: 0.01 },
          });
          series.setData(
            data.timestamps.map((t, i) => ({
              time: t.slice(0, 10) as `${number}-${number}-${number}`,
              value: data.values[i],
            }))
          );
          chart.timeScale().fitContent();
        }}
      />
    </div>
  );
}
