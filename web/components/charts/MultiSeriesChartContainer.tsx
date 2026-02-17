"use client";

import { useRef } from "react";
import type {
  IChartApi,
  ISeriesApi,
  SeriesType,
  DeepPartial,
  ChartOptions,
} from "lightweight-charts";
import ChartContainer from "./ChartContainer";
import { fmtDollar, fmtNumber } from "@/lib/format";

interface SeriesMeta {
  label: string;
  color: string;
}

interface Props {
  seriesMeta: SeriesMeta[];
  onChart: (chart: IChartApi) => ISeriesApi<SeriesType>[];
  valueFormat?: "dollar" | "number";
  options?: DeepPartial<ChartOptions>;
  className?: string;
}

export default function MultiSeriesChartContainer({
  seriesMeta,
  onChart,
  valueFormat = "number",
  options,
  className,
}: Props) {
  const tooltipRef = useRef<HTMLDivElement>(null);

  return (
    <div className="relative">
      <ChartContainer
        options={options}
        className={className}
        onChart={(chart) => {
          const seriesApis = onChart(chart);

          const metaMap = new Map<ISeriesApi<SeriesType>, SeriesMeta>();
          seriesApis.forEach((s, i) => {
            if (seriesMeta[i]) metaMap.set(s, seriesMeta[i]);
          });

          const tooltip = tooltipRef.current;
          if (!tooltip) return;

          const fmt = valueFormat === "dollar" ? fmtDollar : fmtNumber;

          chart.subscribeCrosshairMove((param) => {
            if (!param.point || !param.time || param.seriesData.size === 0) {
              tooltip.style.display = "none";
              return;
            }

            const rows: string[] = [];
            for (const s of seriesApis) {
              const meta = metaMap.get(s);
              const dp = param.seriesData.get(s) as
                | { value: number }
                | undefined;
              if (!meta || dp == null) continue;
              rows.push(
                `<div style="display:flex;align-items:center;gap:6px;white-space:nowrap">` +
                  `<span style="width:8px;height:8px;border-radius:50%;background:${meta.color};flex-shrink:0"></span>` +
                  `<span style="color:#aaa;font-size:11px;flex:1">${meta.label}</span>` +
                  `<span style="color:#fff;font-size:11px;font-weight:600;margin-left:12px">${fmt(dp.value)}</span>` +
                  `</div>`,
              );
            }

            if (rows.length === 0) {
              tooltip.style.display = "none";
              return;
            }

            tooltip.innerHTML = rows.join("");
            tooltip.style.display = "block";
          });
        }}
      />
      <div
        ref={tooltipRef}
        style={{
          display: "none",
          position: "absolute",
          top: 12,
          left: 12,
          background: "rgba(14,17,23,0.92)",
          border: "1px solid rgba(255,255,255,0.12)",
          borderRadius: 6,
          padding: "8px 10px",
          pointerEvents: "none",
          zIndex: 10,
          maxHeight: 320,
          overflowY: "auto",
          backdropFilter: "blur(4px)",
          minWidth: 160,
        }}
      />
    </div>
  );
}
