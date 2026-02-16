# API Route Proxy Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace direct browser-to-Vercel-Blob fetching with server-side API route proxy, eliminating 403 "store blocked" failures.

**Architecture:** Next.js API route (`/api/data/[...path]`) fetches from Vercel Blob using the server-side `BLOB_READ_WRITE_TOKEN`. Frontend SWR hooks hit same-origin `/api/data/...` endpoints. Blob store no longer needs public access. Pages remain statically generated via `generateStaticParams`.

**Tech Stack:** Next.js 16 App Router API Routes, SWR, Vercel Blob REST API

---

### Task 1: Remove static export, add API route

**Files:**
- Modify: `web/next.config.ts`
- Create: `web/app/api/data/[...path]/route.ts`

**Step 1: Remove `output: "export"` from next.config.ts**

```typescript
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  images: { unoptimized: true },
};
export default nextConfig;
```

**Step 2: Create the catch-all API route**

Create `web/app/api/data/[...path]/route.ts`:

```typescript
import { NextRequest, NextResponse } from "next/server";

const BLOB_TOKEN = process.env.BLOB_READ_WRITE_TOKEN ?? "";
const BLOB_STORE_URL = process.env.BLOB_STORE_URL ?? "https://uwo3kkxivibkrwou.public.blob.vercel-storage.com";

const VALID_SYSTEMS = new Set(["oanda", "alpaca", "solana"]);
const VALID_FILES = new Set([
  "overview-daily.json",
  "overview-5min.json",
  "positions.json",
  "risk-daily.json",
  "risk-5min.json",
]);

export async function GET(
  _req: NextRequest,
  { params }: { params: Promise<{ path: string[] }> },
) {
  const segments = (await params).path;

  // Validate path: must be exactly [system, file]
  if (segments.length !== 2) {
    return NextResponse.json({ error: "Not found" }, { status: 404 });
  }

  const [system, file] = segments;
  if (!VALID_SYSTEMS.has(system) || !VALID_FILES.has(file)) {
    return NextResponse.json({ error: "Not found" }, { status: 404 });
  }

  const blobUrl = `${BLOB_STORE_URL}/dashboard/${system}/${file}`;

  try {
    const resp = await fetch(blobUrl, {
      headers: BLOB_TOKEN
        ? { Authorization: `Bearer ${BLOB_TOKEN}` }
        : {},
      next: { revalidate: 30 },
    });

    if (!resp.ok) {
      console.error(`Blob fetch failed: ${resp.status} ${blobUrl}`);
      return NextResponse.json(
        { error: "Data temporarily unavailable" },
        { status: 502 },
      );
    }

    const data = await resp.arrayBuffer();

    return new NextResponse(data, {
      status: 200,
      headers: {
        "Content-Type": "application/json",
        "Cache-Control": "public, s-maxage=30, stale-while-revalidate=60",
      },
    });
  } catch (err) {
    console.error("Blob proxy error:", err);
    return NextResponse.json(
      { error: "Data temporarily unavailable" },
      { status: 502 },
    );
  }
}
```

**Step 3: Verify build succeeds**

Run: `cd web && npx next build`
Expected: Build succeeds, pages still generated via generateStaticParams, new API route available.

---

### Task 2: Update frontend data hooks

**Files:**
- Modify: `web/lib/data.ts`

**Step 1: Rewrite data.ts to use API route with proper error handling**

```typescript
"use client";

import useSWR from "swr";
import type { OverviewData, PositionsData, RiskData, SystemSlug, Timeframe } from "./types";

const fetcher = async (url: string) => {
  const res = await fetch(url);
  if (!res.ok) {
    const error = new Error(`Failed to fetch data (${res.status})`);
    throw error;
  }
  return res.json();
};

const SWR_OPTIONS = {
  revalidateOnFocus: false,
  refreshInterval: 60_000,
  dedupingInterval: 30_000,
  errorRetryCount: 3,
  errorRetryInterval: 5_000,
};

export function useOverview(system: SystemSlug, timeframe: Timeframe) {
  const freq = timeframe === "daily" ? "daily" : "5min";
  return useSWR<OverviewData>(
    `/api/data/${system}/overview-${freq}.json`,
    fetcher,
    SWR_OPTIONS,
  );
}

export function usePositions(system: SystemSlug) {
  return useSWR<PositionsData>(
    `/api/data/${system}/positions.json`,
    fetcher,
    SWR_OPTIONS,
  );
}

export function useRisk(system: SystemSlug, timeframe: Timeframe) {
  const freq = timeframe === "daily" ? "daily" : "5min";
  return useSWR<RiskData>(
    `/api/data/${system}/risk-${freq}.json`,
    fetcher,
    SWR_OPTIONS,
  );
}
```

Key changes from old version:
- Removed `BLOB_BASE` / `NEXT_PUBLIC_BLOB_BASE_URL` (no longer needed)
- Fetcher checks `res.ok` and throws on error (prevents silent JSON parse failures)
- Added `errorRetryCount: 3` (was unlimited — caused the 403 flood)
- Added `errorRetryInterval: 5_000` (5s between retries)

---

### Task 3: Simplify root page redirect

**Files:**
- Modify: `web/app/page.tsx`

**Step 1: Convert from client-side redirect to server-side**

```typescript
import { redirect } from "next/navigation";

export default function Home() {
  redirect("/oanda");
}
```

No longer need `"use client"` or `useEffect` — server redirect works now that we're not using static export.

---

### Task 4: Set env var on Vercel and deploy

**Step 1: Set `BLOB_READ_WRITE_TOKEN` as server-side env var on Vercel**

Run: `vercel env add BLOB_READ_WRITE_TOKEN production`
Value: (from ~/.env.dashboard)

**Step 2: Also set `BLOB_STORE_URL` env var**

Run: `vercel env add BLOB_STORE_URL production`
Value: `https://uwo3kkxivibkrwou.public.blob.vercel-storage.com`

**Step 3: Deploy**

Run: `npx vercel --prod`

**Step 4: Verify in incognito**

- Navigate to `https://avd-trading-dashboard.vercel.app/oanda`
- All tabs should load data
- No 403 errors in console
- API route returns proper JSON with cache headers

---

### Task 5: Clean up old env vars

**Step 1: Remove `NEXT_PUBLIC_BLOB_BASE_URL` from Vercel**

This env var is no longer needed since frontend no longer fetches directly from blob.

Run: `vercel env rm NEXT_PUBLIC_BLOB_BASE_URL production`

**Step 2: Remove from `.env.example` if present**

**Step 3: Commit all changes**
