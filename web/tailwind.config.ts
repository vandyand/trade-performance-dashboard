import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        bg: "#0E1117",
        card: "#1E2130",
        primary: "#4A90D9",
        accent: {
          green: "#00C853",
          red: "#FF1744",
          orange: "#FF9100",
          purple: "#AA00FF",
        },
      },
    },
  },
  plugins: [],
};
export default config;
