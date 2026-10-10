import Link from "next/link";
import { buttonVariants } from "@/components/ui/button";

export default function NotFound() {
  return (
    <div className="max-w-2xl space-y-4">
      <h1 className="text-2xl font-semibold tracking-tight">Page not found</h1>
      <p className="text-sm text-muted-foreground">There is no page at this address.</p>
      <div className="flex flex-wrap gap-2">
        <Link href="/" className={buttonVariants()}>
          Dashboard
        </Link>
        <Link href="/runs" className={buttonVariants({ variant: "outline" })}>
          Runs
        </Link>
        <Link href="/datasets" className={buttonVariants({ variant: "outline" })}>
          Datasets
        </Link>
      </div>
    </div>
  );
}
