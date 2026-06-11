import { SystemSlug } from "./types";

export interface SystemConfig {
  slug: SystemSlug;
  name: string;
  label: string;
  valueName: string;
  description: string[];
  startCaption: string | null;
}

export const SYSTEMS: Record<SystemSlug, SystemConfig> = {
  oanda: {
    slug: "oanda",
    name: "OANDA Forex",
    label: "OANDA",
    valueName: "NAV",
    description: [
      "20 forex instruments",
      "TCN + Actor-Critic RL",
      "Daily position sizing",
      "Daily decision intervals",
    ],
    startCaption: "Traded live Jan 19 – Mar 10, 2026",
  },
  alpaca: {
    slug: "alpaca",
    name: "Alpaca Equities",
    label: "Alpaca",
    valueName: "Equity",
    description: [
      "100 long/short positions",
      "US equities universe",
      "Paper trading",
      "Daily rebalancing",
    ],
    startCaption: "Traded live Feb 2 – Mar 11, 2026, starting balance $100,000.00",
  },
  solana: {
    slug: "solana",
    name: "Solana Altmemecoins",
    label: "Solana",
    valueName: "NAV",
    description: [
      "Solana memecoins (long-only)",
      "TD3 reinforcement learning",
      "5-minute decision intervals",
      "Jupiter DEX execution",
    ],
    startCaption: null,
  },
  kalshi: {
    slug: "kalshi",
    name: "Kalshi Predictions",
    label: "Kalshi",
    valueName: "NAV",
    description: [
      "Deterministic rules engine",
      "Binary prediction contracts",
      "Backtested against historical inefficiencies",
      "5-minute sampling intervals",
    ],
    startCaption: "Traded live Feb 23 – Mar 11, 2026, starting with $5,000.00",
  },
};

export const SYSTEM_SLUGS: SystemSlug[] = ["oanda", "alpaca", "solana", "kalshi"];
