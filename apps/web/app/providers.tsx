"use client";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ThemeProvider } from "next-themes";
import { useEffect, useState, type ReactNode } from "react";
import { Toaster } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";

const MOCK = process.env.NEXT_PUBLIC_API_MOCK === "1";

export function Providers({ children }: { children: ReactNode }) {
  const [client] = useState(
    () => new QueryClient({ defaultOptions: { queries: { retry: 1, staleTime: 2000 } } }),
  );
  const [ready, setReady] = useState(!MOCK);
  const [mockError, setMockError] = useState<string | null>(null);

  useEffect(() => {
    if (!MOCK) return;
    let cancelled = false;
    import("@/mocks/browser")
      .then(({ worker }) => worker.start({ onUnhandledRequest: "bypass", quiet: true }))
      .then(() => {
        if (!cancelled) setReady(true);
      })
      .catch((e: unknown) => {
        if (!cancelled) setMockError(e instanceof Error ? e.message : String(e));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <ThemeProvider attribute="class" defaultTheme="system" enableSystem disableTransitionOnChange>
      <QueryClientProvider client={client}>
        <TooltipProvider>{mockError ? (
            <div role="alert" className="m-4 rounded-md border border-destructive p-4 text-destructive">
              Could not start the mock API (NEXT_PUBLIC_API_MOCK=1): {mockError}
            </div>
          ) : ready ? (
            children
          ) : null}</TooltipProvider>
        <Toaster />
      </QueryClientProvider>
    </ThemeProvider>
  );
}
