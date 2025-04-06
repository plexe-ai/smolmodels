"""
MLFlow callback for tracking model building process.

This module provides a callback implementation that logs model building
metrics, parameters, and artifacts to MLFlow.
"""

import logging
from pathlib import Path
from typing import Optional

from smolmodels.callbacks import Callback, BuildStateInfo
from smolmodels.internal.models.entities.metric import Metric

logger = logging.getLogger(__name__)


class MLFlowCallback(Callback):
    """
    Callback that logs the model building process to MLFlow.

    This callback hooks into the model building process and logs metrics,
    parameters, and artifacts to MLFlow for tracking and visualization.
    """

    def __init__(self, tracking_uri: str, experiment_name: str, connect_timeout: int = 10):
        """
        Initialize MLFlow callback.

        :param tracking_uri: Optional MLFlow tracking server URI.
        :param experiment_name: Name for the MLFlow experiment. Defaults to "smolmodels".
        :param connect_timeout: Timeout in seconds for MLFlow server connection. Defaults to 10.
        """
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.experiment_id = None
        self.connect_timeout = connect_timeout

        # Import MLFlow module
        try:
            import mlflow

            # Clean up active runs, if any
            if mlflow.active_run():
                mlflow.end_run()

            self.mlflow = mlflow

        except Exception as e:
            raise RuntimeError(f"❌  Error importing MLFlow: {e}") from e

        # Set up MLFlow tracking URI and experiment
        try:
            # Set connection timeout for API calls
            import os

            os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = str(self.connect_timeout)
            mlflow.set_tracking_uri(tracking_uri)
            logger.info(f"✅  MLFlow configured with tracking URI '{tracking_uri}'")

        except Exception as e:
            raise RuntimeError(f"❌  Error setting up MLFlow: {e}") from e

    def on_build_start(self, info: BuildStateInfo) -> None:
        """
        Start MLFlow parent run and log initial parameters.

        :param info: Information about the model building process start.
        """
        # Set or get experiment
        experiment = self.mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            self.experiment_id = self.mlflow.create_experiment(self.experiment_name)
        else:
            self.experiment_id = experiment.experiment_id
        logger.info(f"✅  MLFlow configured with experiment '{self.experiment_name}' (ID: {self.experiment_id})")
        # TODO: Start an MLFlow parent run

    def on_build_end(self, info: BuildStateInfo) -> None:
        """
        Log final model details and end MLFlow parent run.

        :param info: Information about the model building process end.
        """
        try:
            if self.mlflow.active_run():
                self.mlflow.end_run()
        except Exception as e:
            raise RuntimeError(f"❌  Error cleaning up MLFlow run: {e}") from e

    def on_iteration_start(self, info: BuildStateInfo) -> None:
        """
        Start a new child run for this iteration if using nested runs.

        :param info: Information about the iteration start.
        """
        run_name = f"iteration-{info.iteration}"
        self.mlflow.start_run(
            run_name=run_name,
            experiment_id=self.experiment_id,
        )
        logger.info(f"✅  Started MLFlow run: {run_name}")

        # Log model parameters
        self.mlflow.log_params(
            {
                "intent": info.intent,
                "input_schema": str(info.input_schema.model_fields),
                "output_schema": str(info.output_schema.model_fields),
                "provider": str(info.provider),
                "run_timeout": info.run_timeout,
                "max_iterations": info.max_iterations,
                "timeout": info.timeout,
                "iteration": info.iteration,
            }
        )

    def on_iteration_end(self, info: BuildStateInfo) -> None:
        """
        Log metrics for this iteration.

        :param info: Information about the iteration end.
        """
        if not self.mlflow.active_run():
            return

        if info.node.training_code:
            try:
                # Save code to a file first, then log it
                code_path = Path("trainer_source.py")
                with open(code_path, "w") as f:
                    f.write(info.node.training_code)
                self.mlflow.log_artifact(str(code_path))
            except Exception as e:
                logger.warning(f"Could not log trainer source: {e}")

        # Log node performance if available
        if info.node.performance:
            self._log_metric(info.node.performance, step=info.iteration)

        # Log execution time
        if info.node.execution_time:
            self.mlflow.log_metric("execution_time", info.node.execution_time, step=info.iteration)

        # Log whether exception was raised
        if info.node.exception_was_raised:
            self.mlflow.set_tag(f"iteration_{info.iteration}_error", str(info.node.exception))

        # Log model artifacts if any
        if info.node.model_artifacts:
            for artifact in info.node.model_artifacts:
                if Path(artifact).exists():
                    try:
                        self.mlflow.log_artifact(str(artifact))
                    except Exception as e:
                        logger.warning(f"Could not log artifact {artifact}: {e}")

        try:
            self.mlflow.end_run()
        except Exception as e:
            logger.warning(f"Error ending MLFlow run: {e}")

    def _log_metric(self, metric: Metric, prefix: str = "", step: Optional[int] = None) -> None:
        """
        Log a SmolModels Metric object to MLFlow.

        :param metric: SmolModels Metric object
        :param prefix: Optional prefix for the metric name
        :param step: Optional step number
        """
        if self.mlflow is None or not self.mlflow.active_run():
            return

        if metric and hasattr(metric, "name") and hasattr(metric, "value"):
            try:
                value = float(metric.value)
                self.mlflow.log_metric(f"{prefix}{metric.name}", value, step=step)
            except (ValueError, TypeError) as e:
                logger.warning(f"Could not convert metric {metric.name} value to float: {e}")
                # Try to log as tag instead
                self.mlflow.set_tag(f"{prefix}{metric.name}", str(metric.value))
