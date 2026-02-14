"use client";

import { useRef, useEffect } from "react";
import { createChart, type IChartApi, type DeepPartial, type ChartOptions } from "lightweight-charts";

const CHART_DEFAULTS: DeepPartial<ChartOptions> = {
  layout: {
    background: { color: "#0E1117" },
    textColor: "#FAFAFA",
    fontFamily: "Inter, system-ui, sans-serif",
  },
  grid: {
    vertLines: { color: "rgba(255,255,255,0.05)" },
    horzLines: { color: "rgba(255,255,255,0.05)" },
  },
  crosshair: {
    vertLine: { color: "rgba(74,144,217,0.3)" },
    horzLine: { color: "rgba(74,144,217,0.3)" },
  },
  timeScale: {
    borderColor: "rgba(255,255,255,0.1)",
    timeVisible: true,
  },
  rightPriceScale: {
    borderColor: "rgba(255,255,255,0.1)",
  },
};

interface Props {
  options?: DeepPartial<ChartOptions>;
  className?: string;
  onChart: (chart: IChartApi) => void;
}

export default function ChartContainer({ options, className, onChart }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    const chart = createChart(containerRef.current, {
      ...CHART_DEFAULTS,
      ...options,
      width: containerRef.current.clientWidth,
      height: 350,
    });

    chartRef.current = chart;
    onChart(chart);

    const resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        chart.applyOptions({ width: entry.contentRect.width });
      }
    });
    resizeObserver.observe(containerRef.current);

    return () => {
      resizeObserver.disconnect();
      chart.remove();
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  return <div ref={containerRef} className={className ?? "w-full"} />;
}
