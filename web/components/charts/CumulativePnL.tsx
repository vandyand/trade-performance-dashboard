"use client";

import { LineSeries } from "lightweight-charts";
import ChartContainer from "./ChartContainer";
import { toChartTime } from "@/lib/chartTime";

const PALETTE = [
  "#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854",
  "#ffd92f", "#e5c494", "#b3b3b3", "#e41a1c", "#377eb8",
  "#4daf4a", "#984ea3", "#ff7f00", "#ffff33", "#a65628",
  "#f781bf", "#999999", "#66c2a5", "#fc8d62", "#8da0cb",
];

interface Props {
  timestamps: string[];
  series: Record<string, number[]>;
  title: string;
}

export default function CumulativePnL({ timestamps, series, title }: Props) {
  const symbols = Object.keys(series);
  if (symbols.length === 0) return null;

  return (
    <div>
      <h3 className="text-sm font-medium text-white/70 mb-2">{title}</h3>
      <ChartContainer
        options={{ height: 450 } as any}
        onChart={(chart) => {
          symbols.forEach((sym, i) => {
            const line = chart.addSeries(LineSeries, {
              color: PALETTE[i % PALETTE.length],
              lineWidth: 2,
              title: sym,
            });
            line.setData(
              timestamps.map((t, j) => ({
                time: toChartTime(t),
                value: series[sym][j] ?? 0,
              }))
            );
          });
          chart.timeScale().fitContent();
        }}
      />
    </div>
  );
}
