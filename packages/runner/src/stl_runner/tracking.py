"""Log job reports to MLflow when MLFLOW_TRACKING_URI is set."""

from __future__ import annotations

import os

from stl_core.schemas import JobSpec, MetricReport


def log_report(report: MetricReport, job: JobSpec,
               experiment: str = "synth-trust-lab") -> str | None:
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        return None
    import mlflow

    mlflow.set_experiment(experiment)
    tags = {"run_id": job.run_id, "dataset_id": job.dataset_id, "status": report.status}
    with mlflow.start_run(run_name=f"{job.run_id}/{job.job_id}", tags=tags) as run:
        mlflow.log_params({"generator": job.generator, "seed": job.seed,
                           "dataset": job.dataset_id,
                           **{f"param.{k}": v for k, v in job.params.items()}})
        mlflow.log_metrics({k: v for k, v in report.flat().items() if v is not None})
        mlflow.log_dict(report.model_dump(mode="json"), "report.json")
        return run.info.run_id
