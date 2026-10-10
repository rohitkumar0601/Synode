import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { FamilyBadge } from "@/components/family-badge";
import { StatusBadge } from "@/components/status-badge";

describe("badges", () => {
  it("labels the audit family as not DP", () => {
    render(<FamilyBadge family="nominal_dp_audit" />);
    expect(screen.getByText("audit — not DP")).toBeInTheDocument();
  });
  it("shows a GPU badge for gpu-only generators", () => {
    render(<FamilyBadge family="diffusion" cpuOk={false} />);
    expect(screen.getByText("GPU")).toBeInTheDocument();
  });
  it("status badge shows text, not only colour", () => {
    render(<StatusBadge state="failed" />);
    expect(screen.getByText("Failed")).toBeInTheDocument();
  });
});
