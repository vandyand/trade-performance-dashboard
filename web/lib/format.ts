export function fmtPct(value: number, decimals = 2): string {
  return `${(value * 100).toFixed(decimals)}%`;
}

export function fmtDollar(value: number, decimals = 2): string {
  const sign = value >= 0 ? "" : "-";
  return `${sign}$${Math.abs(value).toFixed(decimals)}`;
}

export function fmtNumber(value: number, decimals = 2): string {
  return value.toFixed(decimals);
}

export function fmtSignedDollar(value: number, decimals = 2): string {
  const sign = value >= 0 ? "+" : "-";
  return `${sign}$${Math.abs(value).toFixed(decimals)}`;
}
