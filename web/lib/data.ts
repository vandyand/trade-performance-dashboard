"use client";

import useSWR from "swr";
import type { OverviewData, PositionsData, RiskData, SystemSlug, Timeframe } from "./types";

const BLOB_BASE = process.env.NEXT_PUBLIC_BLOB_BASE_URL || "";

const fetcher = (url: string) => fetch(url).then((r) => r.json());

const SWR_OPTIONS = {
  revalidateOnFocus: false,
  refreshInterval: 60_000,
  dedupingInterval: 30_000,
};

export function useOverview(system: SystemSlug, timeframe: Timeframe) {
  const freq = timeframe === "daily" ? "daily" : "5min";
  return useSWR<OverviewData>(
    `${BLOB_BASE}/dashboard/${system}/overview-${freq}.json`,
    fetcher,
    SWR_OPTIONS,
  );
}

export function usePositions(system: SystemSlug) {
  return useSWR<PositionsData>(
    `${BLOB_BASE}/dashboard/${system}/positions.json`,
    fetcher,
    SWR_OPTIONS,
  );
}

export function useRisk(system: SystemSlug, timeframe: Timeframe) {
  const freq = timeframe === "daily" ? "daily" : "5min";
  return useSWR<RiskData>(
    `${BLOB_BASE}/dashboard/${system}/risk-${freq}.json`,
    fetcher,
    SWR_OPTIONS,
  );
}
