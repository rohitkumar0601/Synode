import { fileURLToPath } from "node:url";
import { defineConfig } from "vitest/config";

export default defineConfig({
  oxc: { jsx: { runtime: "automatic" } },
  resolve: { alias: { "@": fileURLToPath(new URL("./", import.meta.url)) } },
  test: {
    environment: "jsdom",
    setupFiles: ["./vitest.setup.ts"],
    include: ["__tests__/**/*.test.{ts,tsx}"],
    globals: false,
    // A loaded machine (parallel suites, OneDrive I/O) can take seconds per render; time-outs are
    // safety nets only, the tests themselves wait on conditions.
    testTimeout: 30000,
  },
});
