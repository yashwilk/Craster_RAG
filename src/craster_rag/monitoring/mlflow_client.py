"""
What MLflow tracks:
    ingestion runs        chunk size overlap model used
    query runs            latency tokens confidence
    experiment comparison which settings work best

MLflow UI:
    run: make mlflow-ui
    open: http://localhost:5000

"""


import logging

import mlflow

from config import settings

logger = logging.getLogger(__name__)


class MLflowTracker:
    def __init__(self):
        self._enabled = False
        self._setup()

    def _setup(self) -> None:
        """Initialise MLflow tracking."""
        try:
            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
            mlflow.set_experiment(settings.mlflow_experiment_name)
            self._enabled = True
            logger.info(
                f"MLflow tracking enabled — "
                f"uri={settings.mlflow_tracking_uri}"
            )
        except Exception as e:
            logger.warning(f"MLflow setup failed: {e}")

    def log_ingestion_params(self, params: dict) -> None:
        if not self._enabled:
            return
        try:
            mlflow.log_params(params)
        except Exception as e:
            logger.warning(f"MLflow params failed: {e}")

    def log_ingestion_metrics(self, metrics: dict) -> None:
        if not self._enabled:
            return
        try:
            mlflow.log_metrics(metrics)
        except Exception as e:
            logger.warning(f"MLflow query log failed: {e}")

    def start_run(self, run_name: str) -> None:
        """Start a new MLflow run."""
        if not self._enabled:
            return
        try:
            mlflow.start_run(run_name=run_name)
        except Exception as e:
            logger.warning(f"MLflow start_run failed: {e}")

    def end_run(self) -> None:
        """End the current MLflow run."""
        if not self._enabled:
            return
        try:
            mlflow.end_run()
        except Exception as e:
            logger.warning(f"MLflow end_run failed: {e}")


mlflow_tracker = MLflowTracker()
