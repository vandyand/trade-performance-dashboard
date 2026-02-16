import type { UTCTimestamp, BusinessDay } from "lightweight-charts";

/**
 * Convert an ISO timestamp to a lightweight-charts time value.
 * - Daily-looking timestamps (one per day) → "YYYY-MM-DD" business day string
 * - Intraday timestamps → Unix seconds (UTCTimestamp)
 */
export function toChartTime(iso: string): UTCTimestamp | `${number}-${number}-${number}` {
  return Math.floor(new Date(iso).getTime() / 1000) as UTCTimestamp;
}
