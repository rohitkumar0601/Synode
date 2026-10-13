# Next.js web app (build context = apps/web; ignores in docker/web.Dockerfile.dockerignore).
# NEXT_PUBLIC_API_URL is inlined into the client bundle at build time.
FROM node:24-bookworm-slim AS build
WORKDIR /app
COPY package.json package-lock.json ./
# package-lock.json is generated on Windows and can miss Linux-only optional deps (e.g.
# @emnapi/* for the wasm fallbacks): npm ci rejects it, so fall back to npm install, which
# keeps every locked version and only adds the missing entries (inside the image only).
RUN --mount=type=cache,target=/root/.npm npm ci --no-audit --no-fund \
    || npm install --no-audit --no-fund
COPY . .
ARG NEXT_PUBLIC_API_URL=http://localhost:8010
ENV NEXT_PUBLIC_API_URL=${NEXT_PUBLIC_API_URL} NEXT_TELEMETRY_DISABLED=1
# --no-audit: the audit round-trip made prune take ~40 min on the dev machine.
RUN npm run build && npm prune --omit=dev --no-audit --no-fund

FROM node:24-bookworm-slim
WORKDIR /app
ENV NODE_ENV=production NEXT_TELEMETRY_DISABLED=1
COPY --from=build --chown=node:node /app/package.json /app/next.config.ts ./
COPY --from=build --chown=node:node /app/node_modules node_modules
COPY --from=build --chown=node:node /app/.next .next
COPY --from=build --chown=node:node /app/public public
USER node
EXPOSE 3010
CMD ["node_modules/.bin/next", "start", "-p", "3010"]
