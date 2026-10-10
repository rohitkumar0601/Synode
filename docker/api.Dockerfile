# API image (build context = repo root): FastAPI + Alembic migrations + builtin dataset confs.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS build
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv UV_HTTP_TIMEOUT=300 UV_HTTP_RETRIES=5 UV_CONCURRENT_DOWNLOADS=4
WORKDIR /src
COPY packages packages
COPY services/api/pyproject.toml services/api/uv.lock services/api/
COPY services/api/src services/api/src
RUN --mount=type=cache,id=uv-api,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable --project services/api

FROM python:3.12-slim-bookworm
RUN useradd --create-home --uid 1000 stl \
    && mkdir /stl-store \
    && chown stl /stl-store
COPY --from=build /opt/venv /opt/venv
WORKDIR /app
COPY services/api/alembic.ini services/api/alembic.ini
COPY services/api/migrations services/api/migrations
COPY experiments/conf conf
ENV PATH=/opt/venv/bin:$PATH PYTHONUNBUFFERED=1 STL_CONF_DIR=/app/conf
USER stl
EXPOSE 8000
CMD ["sh", "-c", "alembic -c services/api/alembic.ini upgrade head && exec uvicorn stl_api.main:app --host 0.0.0.0 --port 8000"]
