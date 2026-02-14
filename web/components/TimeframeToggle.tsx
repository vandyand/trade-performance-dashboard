"use client";

import type { Timeframe } from "@/lib/types";

interface Props {
  value: Timeframe;
  onChange: (tf: Timeframe) => void;
}

export default function TimeframeToggle({ value, onChange }: Props) {
  return (
    <div className="inline-flex rounded-lg bg-card p-1 gap-1">
      {(["daily", "5min"] as const).map((tf) => (
        <button
          key={tf}
          onClick={() => onChange(tf)}
          className={`px-4 py-1.5 rounded-md text-sm font-medium transition-colors ${
            value === tf
              ? "bg-primary text-white"
              : "text-white/60 hover:text-white hover:bg-white/5"
          }`}
        >
          {tf === "daily" ? "Daily" : "5-Minute"}
        </button>
      ))}
    </div>
  );
}
