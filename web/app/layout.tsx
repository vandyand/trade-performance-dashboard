import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import SystemSidebar from "@/components/SystemSidebar";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Trading Performance Dashboard",
  description: "Algorithmic trading system metrics",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body className={`${inter.className} min-h-screen flex`}>
        <SystemSidebar />
        <main className="flex-1 overflow-auto">{children}</main>
      </body>
    </html>
  );
}
