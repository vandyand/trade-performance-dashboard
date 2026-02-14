interface Column {
  key: string;
  label: string;
  format?: (v: any) => string;
  align?: "left" | "right";
}

interface Props {
  columns: Column[];
  rows: Record<string, any>[];
  title?: string;
}

export default function DataTable({ columns, rows, title }: Props) {
  if (rows.length === 0) return null;

  return (
    <div>
      {title && (
        <h3 className="text-sm font-medium text-white/70 mb-2">{title}</h3>
      )}
      <div className="overflow-x-auto rounded-lg border border-white/10">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-white/10 bg-card">
              {columns.map((col) => (
                <th
                  key={col.key}
                  className={`px-3 py-2 font-medium text-white/50 ${
                    col.align === "right" ? "text-right" : "text-left"
                  }`}
                >
                  {col.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => (
              <tr key={i} className="border-b border-white/5 hover:bg-white/5">
                {columns.map((col) => (
                  <td
                    key={col.key}
                    className={`px-3 py-1.5 ${
                      col.align === "right" ? "text-right" : "text-left"
                    }`}
                  >
                    {col.format ? col.format(row[col.key]) : row[col.key]}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
