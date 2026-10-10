import type { Metadata } from "next";
import { Nav } from "@/components/nav";
import { Providers } from "./providers";
import "./globals.css";

export const metadata: Metadata = {
  title: "Synthetic Data Trust Lab",
  description: "Benchmark synthetic-data generators and get a privacy-gated recommendation.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="min-h-screen bg-background text-foreground antialiased">
        <Providers>
          <a
            href="#main"
            className="sr-only focus:not-sr-only focus:absolute focus:left-2 focus:top-2 focus:rounded-md focus:bg-background focus:px-3 focus:py-2"
          >
            Skip to content
          </a>
          <Nav />
          <main id="main" className="mx-auto max-w-6xl px-4 py-8">
            {children}
          </main>
        </Providers>
      </body>
    </html>
  );
}
