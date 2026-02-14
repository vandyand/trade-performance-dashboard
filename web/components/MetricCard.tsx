interface Props {
  label: string;
  value: string;
}

export default function MetricCard({ label, value }: Props) {
  return (
    <div className="bg-card rounded-xl p-4">
      <p className="text-xs text-white/50 uppercase tracking-wider">{label}</p>
      <p className="text-2xl font-semibold mt-1">{value}</p>
    </div>
  );
}
