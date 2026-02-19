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
    startCaption: "Algorithm began trading Jan 19, 2026",
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
    startCaption: "Algorithm began trading Feb 2, 2026 with account balance of $100,000.00",
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
      "25 specialized trading agents",
      "Binary prediction contracts",
      "Paper trading",
      "5-minute sampling intervals",
    ],
    startCaption: "Algorithm began trading Feb 19, 2026 with $3,750.00 across 25 agents",
  },
};

export const SYSTEM_SLUGS: SystemSlug[] = ["oanda", "alpaca", "solana", "kalshi"];
