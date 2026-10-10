"use client";

import { AlertCircle, RotateCw } from "lucide-react";
import Link from "next/link";
import { useEffect } from "react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button, buttonVariants } from "@/components/ui/button";

/** Route error boundary: a rendering error shows this instead of a blank page. */
export default function RouteError({ error, reset }: { error: Error & { digest?: string }; reset: () => void }) {
  useEffect(() => {
    console.error(error);
  }, [error]);
  return (
    <div className="max-w-2xl space-y-4">
      <Alert variant="destructive">
        <AlertCircle aria-hidden="true" />
        <AlertTitle>Something went wrong on this page</AlertTitle>
        <AlertDescription>
          {error.message || "An unexpected error occurred."}
          {error.digest ? ` (reference ${error.digest})` : ""}
        </AlertDescription>
      </Alert>
      <div className="flex flex-wrap gap-2">
        <Button onClick={() => reset()}>
          <RotateCw aria-hidden="true" />
          Try again
        </Button>
        <Link href="/" className={buttonVariants({ variant: "outline" })}>
          Go to the dashboard
        </Link>
      </div>
    </div>
  );
}
