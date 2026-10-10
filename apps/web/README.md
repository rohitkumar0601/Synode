# Synthetic Data Trust Lab — web app

Next.js (App Router) front end for the Trust Lab: upload a tabular dataset, benchmark synthetic-data generators, and read a privacy-gated recommendation (scorecard, presets, Pareto plot).

## Install and run

```bash
npm install
cp .env.example .env.local   # optional; see the table below
npm run dev                  # http://localhost:3010
```

Production: `npm run build && npm start` (also port 3010).

## Configuration

All variables are `NEXT_PUBLIC_*`, so they are inlined at build time: restart `npm run dev` (or rebuild) after changing them.

| Variable | Default | Meaning |
| --- | --- | --- |
| `NEXT_PUBLIC_API_URL` | `http://127.0.0.1:8010` | Base URL of the FastAPI service |
| `NEXT_PUBLIC_API_MOCK` | unset | `1` serves the API from in-browser MSW mocks (no backend needed) |
| `NEXT_PUBLIC_MAX_UPLOAD_MB` | `200` | Client-side upload size guard. Must match the API's `STL_MAX_UPLOAD_MB` |

### Mock mode

```bash
NEXT_PUBLIC_API_MOCK=1 npm run dev
```

(PowerShell: `$env:NEXT_PUBLIC_API_MOCK = "1"; npm run dev`.) Mock mode ships two datasets, the full generator catalog, a finished Adult run (`adult-done`) and a live run (`adult-live`) whose jobs advance on every poll.

### Real API

Start the API (default `http://127.0.0.1:8010`), then leave `NEXT_PUBLIC_API_MOCK` unset and set `NEXT_PUBLIC_API_URL` if the API is elsewhere. Keep `NEXT_PUBLIC_MAX_UPLOAD_MB` equal to the API's `STL_MAX_UPLOAD_MB`.

**Open the web app at <http://localhost:3010>** — that is the origin the API allows by default (CORS). Opening it as `http://127.0.0.1:3010` (or from another host/port) makes every API call fail in the browser with "Cannot reach the API, or it rejected the request"; to use another origin, add it to the API's `STL_CORS_ORIGINS` and restart the API.

## Scripts

- `npm run dev` — dev server on port 3010
- `npm run build` / `npm start` — production build / server
- `npm run lint` — ESLint
- `npm run typecheck` — `tsc --noEmit`
- `npm test` — Vitest + Testing Library + MSW (node server in `mocks/server.ts`)

## Layout

- `app/` — routes: `/`, `/datasets`, `/datasets/[id]`, `/runs`, `/runs/new`, `/runs/[id]`, `/import`
- `lib/types.ts` — API types mirroring the Python schemas (snake_case kept)
- `lib/api.ts` — typed client; non-2xx throws `ApiError{status, detail}`
- `mocks/` — MSW handlers and fixtures shared by the browser and tests
- `components/ui/` — shadcn/ui primitives
