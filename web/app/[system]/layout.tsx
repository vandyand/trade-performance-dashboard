import { SYSTEM_SLUGS } from "@/lib/systems";
import SystemLayoutClient from "./SystemLayoutClient";
import type { SystemSlug } from "@/lib/types";

export function generateStaticParams() {
  return SYSTEM_SLUGS.map((system) => ({ system }));
}

export default async function SystemLayout({
  children,
  params,
}: {
  children: React.ReactNode;
  params: Promise<{ system: string }>;
}) {
  const { system } = await params;
  return (
    <SystemLayoutClient system={system as SystemSlug}>
      {children}
    </SystemLayoutClient>
  );
}
