"use client";

import { AreaSeries } from "lightweight-charts";
import MultiSeriesChartContainer from "./MultiSeriesChartContainer";
import { toChartTime } from "@/lib/chartTime";

const EXPOSURE_COLORS: Record<string, { line: string; top: string }> = {
  "Long Exposure":    { line: "#00C853", top: "rgba(0, 200, 83, 0.3)" },
  "Short Exposure":   { line: "#FF1744", top: "rgba(255, 23, 68, 0.3)" },
  "Margin Used":      { line: "#FF9100", top: "rgba(255, 145, 0, 0.3)" },
  "Token Positions":  { line: "#00C853", top: "rgba(0, 200, 83, 0.3)" },
  "SOL Balance":      { line: "#4A90D9", top: "rgba(74, 144, 217, 0.3)" },
};

interface Props {
  timestamps: string[];
  series: Record<string, number[]>;
}

export default function ExposureChart({ timestamps, series }: Props) {
  const labels = Object.keys(series);
  if (labels.length === 0) return null;

  return (
    <div>
      <h3 className="text-sm font-medium text-white/70 mb-2">Exposure Over Time</h3>
      <MultiSeriesChartContainer
        valueFormat="dollar"
        seriesMeta={labels.map((label) => ({
          label,
          color: (EXPOSURE_COLORS[label] ?? { line: "#4A90D9" }).line,
        }))}
        onChart={(chart) => {
          const seriesApis = labels.map((label) => {
            const colors = EXPOSURE_COLORS[label] ?? { line: "#4A90D9", top: "rgba(74,144,217,0.3)" };
            const area = chart.addSeries(AreaSeries, {
              lineColor: colors.line,
              topColor: colors.top,
              bottomColor: "transparent",
              lineWidth: 1,
              lastValueVisible: false,
              priceLineVisible: false,
            });
            area.setData(
              timestamps.map((t, i) => ({
                time: toChartTime(t),
                value: series[label][i] ?? 0,
              }))
            );
            return area;
          });
          chart.timeScale().fitContent();
          return seriesApis;
        }}
      />
    </div>
  );
}
