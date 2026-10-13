# Celery worker image, one per library env (build context = repo root):
#   docker build -f docker/worker.Dockerfile --build-arg WORKER=sdv --build-arg PYTHON=3.12 .
# WORKER: sdv (3.12) | synthcity (3.11) | smartnoise (3.12) | eval (3.11) | meta (3.11)
# The env is resolved from workers/<WORKER>/uv.lock (--frozen). Path sources (packages/,
# third_party/) are built as regular wheels (--no-editable), so the runtime stage carries only
# /opt/venv, not the sources. CPU-only: uv_sync_cpu.py skips the CUDA stack and swaps a PyPI
# torch for the CPU build of the same locked version (synthcity already locks CPU torch).
ARG PYTHON=3.12

FROM ghcr.io/astral-sh/uv:python${PYTHON}-bookworm-slim AS build
ARG WORKER
ENV UV_COMPILE_BYTECODE=0 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv UV_HTTP_TIMEOUT=300 UV_HTTP_RETRIES=5 UV_CONCURRENT_DOWNLOADS=4
WORKDIR /src
COPY docker/uv_sync_cpu.py /usr/local/bin/uv_sync_cpu.py
COPY packages packages
COPY workers/${WORKER} workers/${WORKER}
# third_party is bind-mounted read-write (setuptools writes build/ and *.egg-info into the
# source tree; those writes are discarded with the mount).
RUN --mount=type=cache,id=uv-${WORKER},target=/root/.cache/uv \
    --mount=type=bind,source=third_party,target=/src/third_party,rw \
    test -n "${WORKER}" && \
    python /usr/local/bin/uv_sync_cpu.py workers/${WORKER}

FROM python:${PYTHON}-slim-bookworm
ARG WORKER
# libgomp1: OpenMP runtime for torch/lightgbm/xgboost wheels.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 stl \
    && mkdir /stl-store \
    && chown stl /stl-store
COPY --from=build /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH PYTHONUNBUFFERED=1 STL_WORKER=${WORKER} \
    MPLCONFIGDIR=/tmp/matplotlib
USER stl
WORKDIR /home/stl
CMD ["celery", "-A", "stl_runner.celery_app", "worker", "--loglevel=INFO"]
