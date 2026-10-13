# MLflow tracking server: sqlite backend on a volume, artifacts proxied (--serve-artifacts) to
# the S3 store (host MinIO); boto3 is for that server-side S3 upload. Clients log via
# mlflow-artifacts:/, so workers need no S3 credentials for MLflow.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS build
ARG MLFLOW_VERSION=3.16.1
ENV UV_PROJECT_ENVIRONMENT=/opt/venv UV_HTTP_TIMEOUT=300 UV_HTTP_RETRIES=5 UV_CONCURRENT_DOWNLOADS=4
RUN uv venv /opt/venv && uv pip install "mlflow==${MLFLOW_VERSION}" boto3

FROM python:3.12-slim-bookworm
RUN useradd --create-home --uid 1000 stl && mkdir /mlflow && chown stl /mlflow
COPY --from=build /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
USER stl
EXPOSE 5000
