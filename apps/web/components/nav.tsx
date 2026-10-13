"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Menu, X } from "lucide-react";
import { useState } from "react";
import { ThemeToggle } from "@/components/theme-toggle";
import { cn } from "@/lib/utils";

const LINKS = [
  { href: "/", label: "Dashboard" },
  { href: "/datasets", label: "Datasets" },
  { href: "/runs", label: "Runs" },
  { href: "/runs/new", label: "New run" },
  { href: "/import", label: "Import" },
];

export function Nav() {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);
  const active = (href: string) =>
    href === "/"
      ? pathname === "/"
      : href === "/runs"
        ? pathname.startsWith("/runs") && !pathname.startsWith("/runs/new")
        : pathname.startsWith(href);
  return (
    <header className="border-b bg-background">
      <div className="mx-auto flex max-w-6xl flex-wrap items-center gap-x-6 gap-y-2 px-4 py-3">
        <Link href="/" className="font-semibold tracking-tight">
          Synthetic Data Trust Lab
        </Link>
        <div className="ml-auto flex items-center gap-1 sm:order-last">
          <ThemeToggle />
          <button
            type="button"
            className="rounded-md p-2 hover:bg-muted focus-visible:outline-2 focus-visible:outline-ring sm:hidden"
            aria-expanded={open}
            aria-controls="main-nav"
            aria-label={open ? "Close menu" : "Open menu"}
            onClick={() => setOpen((o) => !o)}
          >
            {open ? <X className="size-5" aria-hidden="true" /> : <Menu className="size-5" aria-hidden="true" />}
          </button>
        </div>
        <nav
          id="main-nav"
          aria-label="Main"
          className={cn("w-full flex-col gap-1 sm:flex sm:w-auto sm:flex-row sm:flex-wrap", open ? "flex" : "hidden")}
        >
          {LINKS.map((l) => (
            <Link
              key={l.href}
              href={l.href}
              onClick={() => setOpen(false)}
              aria-current={active(l.href) ? "page" : undefined}
              className={cn(
                "rounded-md px-3 py-1.5 text-sm text-muted-foreground transition-colors hover:bg-muted hover:text-foreground focus-visible:outline-2 focus-visible:outline-ring",
                active(l.href) && "bg-muted font-medium text-foreground",
              )}
            >
              {l.label}
            </Link>
          ))}
        </nav>
      </div>
    </header>
  );
}
