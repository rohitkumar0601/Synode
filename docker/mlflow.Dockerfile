# MLflow tracking server: sqlite backend on a volume, artifacts proxied (--serve-artifacts) to
# the S3 store (host MinIO); boto3 is for that server-side S3 upload. Clients log via
# mlflow-artifacts:/, so workers need no S3 credentials for MLflow.
FROM python:3.12-slim-bookworm
ARG MLFLOW_VERSION=3.16.1
RUN pip install --no-cache-dir "mlflow==${MLFLOW_VERSION}" boto3 \
    && useradd --create-home --uid 1000 stl && mkdir /mlflow && chown stl /mlflow
USER stl
EXPOSE 5000
