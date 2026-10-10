# MLflow tracking server: sqlite backend and proxied artifacts on a local volume.
# Clients upload via mlflow-artifacts:/, so workers do not need direct artifact-path access.
FROM python:3.12-slim-bookworm
ARG MLFLOW_VERSION=3.16.1
RUN pip install --no-cache-dir "mlflow==${MLFLOW_VERSION}" \
    && useradd --create-home --uid 1000 stl && mkdir /mlflow && chown stl /mlflow
USER stl
EXPOSE 5000
