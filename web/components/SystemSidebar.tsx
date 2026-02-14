"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { SYSTEMS, SYSTEM_SLUGS } from "@/lib/systems";
import type { SystemSlug } from "@/lib/types";

function extractSystem(pathname: string): SystemSlug {
  const segment = pathname.split("/")[1];
  if (SYSTEM_SLUGS.includes(segment as SystemSlug)) return segment as SystemSlug;
  return "oanda";
}

export default function SystemSidebar() {
  const pathname = usePathname();
  const current = extractSystem(pathname);
  const config = SYSTEMS[current];

  return (
    <aside className="w-64 shrink-0 border-r border-white/10 p-5 flex flex-col gap-6">
      <div>
        <h2 className="text-xs font-semibold uppercase tracking-wider text-white/50 mb-3">
          Trading System
        </h2>
        <div className="flex flex-col gap-1">
          {SYSTEM_SLUGS.map((slug) => (
            <Link
              key={slug}
              href={`/${slug}`}
              className={`px-3 py-2 rounded-lg text-sm transition-colors ${
                slug === current
                  ? "bg-primary/20 text-primary font-medium"
                  : "text-white/70 hover:bg-white/5 hover:text-white"
              }`}
            >
              {SYSTEMS[slug].name}
            </Link>
          ))}
        </div>
      </div>

      <div className="border-t border-white/10 pt-4">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-white/50 mb-2">
          System Details
        </h3>
        <ul className="text-sm text-white/60 space-y-1">
          {config.description.map((line, i) => (
            <li key={i}>- {line}</li>
          ))}
        </ul>
        {config.startCaption && (
          <p className="text-xs text-white/40 mt-2">{config.startCaption}</p>
        )}
      </div>
    </aside>
  );
}
