"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState, createContext, useContext } from "react";
import TimeframeToggle from "@/components/TimeframeToggle";
import type { SystemSlug, Timeframe } from "@/lib/types";

const TABS = [
  { label: "Overview", href: "" },
  { label: "Positions", href: "/positions" },
  { label: "Risk", href: "/risk" },
];

export const TimeframeContext = createContext<Timeframe>("daily");
export function useTimeframe() {
  return useContext(TimeframeContext);
}

interface Props {
  system: SystemSlug;
  children: React.ReactNode;
}

export default function SystemLayoutClient({ system, children }: Props) {
  const pathname = usePathname();
  const [timeframe, setTimeframe] = useState<Timeframe>("daily");

  const basePath = `/${system}`;

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Algorithmic Trading Performance</h1>
          <p className="text-sm text-white/50">Live system metrics | Updated every 5 minutes</p>
        </div>
        <TimeframeToggle value={timeframe} onChange={setTimeframe} />
      </div>

      <nav className="flex gap-1 border-b border-white/10 pb-px">
        {TABS.map((tab) => {
          const href = `${basePath}${tab.href}`;
          const isActive =
            tab.href === ""
              ? pathname === basePath || pathname === `${basePath}/`
              : pathname.startsWith(href);
          return (
            <Link
              key={tab.label}
              href={href}
              className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
                isActive
                  ? "border-primary text-primary"
                  : "border-transparent text-white/60 hover:text-white"
              }`}
            >
              {tab.label}
            </Link>
          );
        })}
      </nav>

      <TimeframeContext.Provider value={timeframe}>
        {children}
      </TimeframeContext.Provider>
    </div>
  );
}
