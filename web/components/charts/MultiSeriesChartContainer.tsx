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

            // Collect entries with values, then sort by value descending
            const entries: { color: string; label: string; value: number }[] = [];
            for (const s of seriesApis) {
              const meta = metaMap.get(s);
              const dp = param.seriesData.get(s) as
                | { value: number }
                | undefined;
              if (!meta || dp == null) continue;
              entries.push({ color: meta.color, label: meta.label, value: dp.value });
            }

            if (entries.length === 0) {
              tooltip.style.display = "none";
              return;
            }

            entries.sort((a, b) => b.value - a.value);

            const rows = entries.map(
              (e) =>
                `<div style="display:flex;align-items:center;gap:6px;white-space:nowrap">` +
                `<span style="width:8px;height:8px;border-radius:50%;background:${e.color};flex-shrink:0"></span>` +
                `<span style="color:#aaa;font-size:11px;flex:1">${e.label}</span>` +
                `<span style="color:#fff;font-size:11px;font-weight:600;margin-left:12px">${fmt(e.value)}</span>` +
                `</div>`,
            );

            tooltip.innerHTML = rows.join("");
            tooltip.style.display = "block";

            // Position near cursor with overflow protection
            const OFFSET = 16;
            const container = tooltip.parentElement;
            if (container) {
              const cw = container.clientWidth;
              const ch = container.clientHeight;
              const tw = tooltip.offsetWidth;
              const th = tooltip.offsetHeight;

              let x = param.point.x + OFFSET;
              let y = param.point.y + OFFSET;

              // Flip left if overflowing right
              if (x + tw > cw) x = param.point.x - tw - OFFSET;
              // Flip up if overflowing bottom
              if (y + th > ch) y = param.point.y - th - OFFSET;

              tooltip.style.left = `${Math.max(0, x)}px`;
              tooltip.style.top = `${Math.max(0, y)}px`;
            }
          });
        }}
      />
      <div
        ref={tooltipRef}
        style={{
          display: "none",
          position: "absolute",
          top: 0,
          left: 0,
          background: "rgba(14,17,23,0.92)",
          border: "1px solid rgba(255,255,255,0.12)",
          borderRadius: 6,
          padding: "8px 10px",
          pointerEvents: "none",
          zIndex: 10,
          backdropFilter: "blur(4px)",
          minWidth: 160,
        }}
      />
    </div>
  );
}
