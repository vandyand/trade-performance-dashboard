interface BarItem {
  label: string;
  value: number;
}

interface Props {
  items: BarItem[];
  title: string;
  formatValue?: (v: number) => string;
}

export default function HorizontalBar({ items, title, formatValue }: Props) {
  if (items.length === 0) return null;

  const maxAbs = Math.max(...items.map((d) => Math.abs(d.value)), 0.01);
  const fmt = formatValue ?? ((v: number) => `$${v.toFixed(2)}`);

  return (
    <div>
      <h3 className="text-sm font-medium text-white/70 mb-3">{title}</h3>
      <div
        className="space-y-1 overflow-y-auto"
        style={{ maxHeight: Math.max(400, items.length * 28) }}
      >
        {items.map((item) => {
          const pct = (Math.abs(item.value) / maxAbs) * 100;
          const isPositive = item.value >= 0;
          return (
            <div key={item.label} className="flex items-center gap-2 text-xs">
              <span className="w-20 text-right text-white/60 shrink-0 truncate">
                {item.label}
              </span>
              <div className="flex-1 h-5 bg-white/5 rounded relative">
                <div
                  className={`h-full rounded ${isPositive ? "bg-accent-green/70" : "bg-accent-red/70"}`}
                  style={{ width: `${pct}%` }}
                />
              </div>
              <span
                className={`w-20 text-right shrink-0 ${
                  isPositive ? "text-accent-green" : "text-accent-red"
                }`}
              >
                {fmt(item.value)}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
