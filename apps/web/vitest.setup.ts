import "@testing-library/jest-dom/vitest";
import { cleanup, configure } from "@testing-library/react";
import { afterAll, afterEach, beforeAll } from "vitest";
import { resetMockState } from "@/mocks/handlers";
import { server } from "@/mocks/server";

// findBy*/waitFor give up after this long; generous because they return as soon as the condition holds.
configure({ asyncUtilTimeout: 10000 });

process.env.NEXT_PUBLIC_POLL_MS = "40";
process.env.NEXT_PUBLIC_API_URL = "http://api.test";

beforeAll(() => server.listen({ onUnhandledRequest: "error" }));
afterEach(() => {
  cleanup();
  server.resetHandlers();
  resetMockState();
});
afterAll(() => server.close());
