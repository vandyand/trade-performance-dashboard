"use client";

import useSWR from "swr";
import type { OverviewData, PositionsData, RiskData, SystemSlug, Timeframe } from "./types";

const fetcher = async (url: string) => {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Failed to fetch data (${res.status})`);
  }
  return res.json();
};

const SWR_OPTIONS = {
  revalidateOnFocus: false,
  refreshInterval: 60_000,
  dedupingInterval: 30_000,
  errorRetryCount: 3,
  errorRetryInterval: 5_000,
};

export function useOverview(system: SystemSlug, timeframe: Timeframe) {
  const freq = timeframe === "daily" ? "daily" : "5min";
  return useSWR<OverviewData>(
    `/data/${system}/overview-${freq}.json`,
    fetcher,
    SWR_OPTIONS,
  );
}

export function usePositions(system: SystemSlug) {
  return useSWR<PositionsData>(
    `/data/${system}/positions.json`,
    fetcher,
    SWR_OPTIONS,
  );
}

export function useRisk(system: SystemSlug, timeframe: Timeframe) {
  const freq = timeframe === "daily" ? "daily" : "5min";
  return useSWR<RiskData>(
    `/data/${system}/risk-${freq}.json`,
    fetcher,
    SWR_OPTIONS,
  );
}
