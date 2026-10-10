import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import RouteError from "@/app/error";
import NotFound from "@/app/not-found";

describe("route error boundary", () => {
  afterEach(() => vi.restoreAllMocks());

  it("shows the error and retries", async () => {
    const logged = vi.spyOn(console, "error").mockImplementation(() => {});
    const reset = vi.fn();
    const user = userEvent.setup();
    render(<RouteError error={new Error("Cannot read properties of undefined (reading 'columns')")} reset={reset} />);
    expect(screen.getByRole("alert")).toHaveTextContent("Something went wrong on this page");
    expect(screen.getByRole("alert")).toHaveTextContent("reading 'columns'");
    await user.click(screen.getByRole("button", { name: "Try again" }));
    expect(reset).toHaveBeenCalledTimes(1);
    expect(screen.getByRole("link", { name: "Go to the dashboard" })).toHaveAttribute("href", "/");
    expect(logged).toHaveBeenCalled();
  });
});

describe("not-found page", () => {
  it("links back into the app", () => {
    render(<NotFound />);
    expect(screen.getByRole("heading", { name: "Page not found" })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Dashboard" })).toHaveAttribute("href", "/");
  });
});
