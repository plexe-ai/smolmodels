"""
MLFlow callback for tracking model building process.

This module provides a callback implementation that logs model building
metrics, parameters, and artifacts to MLFlow.
"""

import logging
from pathlib import Path
from typing import Optional

from smolmodels.callbacks import Callback, BuildStartInfo, BuildEndInfo, IterationStartInfo, IterationEndInfo
from smolmodels.internal.models.entities.metric import Metric

logger = logging.getLogger(__name__)


class MLFlowCallback(Callback):
    """
    Callback that logs the model building process to MLFlow.

    This callback hooks into the model building process and logs metrics,
    parameters, and artifacts to MLFlow for tracking and visualization.
    """

    def __init__(
        self, tracking_uri: str, experiment_name: str, nested: bool = True, parent_run_id: Optional[str] = None
    ):
        """
        Initialize MLFlow callback.

        :param tracking_uri: Optional MLFlow tracking server URI.
        :param experiment_name: Name for the MLFlow experiment. Defaults to "smolmodels".
        :param nested: Whether to create nested runs for iterations. Defaults to True.
        :param parent_run_id: Optional parent run ID for nesting within an existing run.
        """
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.nested = nested
        self.parent_run_id = parent_run_id
        self.experiment_id = None

        self.mlflow = self._import_mlflow()
        if self.mlflow:
            self._setup_mlflow()

    @staticmethod
    def _import_mlflow():
        """Import MLFlow module if available."""
        try:
            import mlflow

            return mlflow
        except ImportError:
            logger.warning("MLFlow is not installed. Install it with 'pip install mlflow' to use this callback.")
            return None

    def _setup_mlflow(self) -> None:
        """Configure MLFlow tracking URI and experiment."""
        if self.mlflow is None:
            return

        if self.tracking_uri:
            self.mlflow.set_tracking_uri(self.tracking_uri)

        # Set or get experiment
        experiment = self.mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            self.experiment_id = self.mlflow.create_experiment(self.experiment_name)
        else:
            self.experiment_id = experiment.experiment_id

        logger.info(f"MLFlow configured with experiment '{self.experiment_name}' (ID: {self.experiment_id})")

    def on_build_start(self, info: BuildStartInfo) -> None:
        """
        Start MLFlow parent run and log initial parameters.

        :param info: Information about the model building process start.
        """
        if self.mlflow is None:
            return

        run_name = f"model-{info.identifier}"

        if self.parent_run_id:
            # Use the provided parent run ID
            self.mlflow.start_run(run_id=self.parent_run_id, nested=True)
            logger.info(f"Resuming MLFlow parent run: {self.parent_run_id}")
        else:
            # Start a new run
            self.mlflow.start_run(experiment_id=self.experiment_id, run_name=run_name)
            self.parent_run_id = self.mlflow.active_run().info.run_id
            logger.info(f"Started MLFlow parent run: {self.parent_run_id}")

        # Log model parameters
        params = {
            "intent": info.intent,
            "provider": str(info.provider),
            "run_timeout": info.run_timeout,
            "max_iterations": info.max_iterations,
            "timeout": info.timeout,
        }

        # Log input and output schemas if available
        if info.input_schema:
            try:
                params["input_schema"] = str(info.input_schema.model_json_schema())
            except Exception as e:
                logger.warning(f"Could not log input schema: {e}")

        if info.output_schema:
            try:
                params["output_schema"] = str(info.output_schema.model_json_schema())
            except Exception as e:
                logger.warning(f"Could not log output schema: {e}")

        # Log constraints if any
        if info.constraints:
            for i, constraint in enumerate(info.constraints):
                params[f"constraint_{i}"] = str(constraint)

        self.mlflow.log_params(params)

    def on_build_end(self, info: BuildEndInfo) -> None:
        """
        Log final model details and end MLFlow parent run.

        :param info: Information about the model building process end.
        """
        if self.mlflow is None or not self.mlflow.active_run():
            return

        # Log final state
        self.mlflow.set_tag("model_state", info.state.value)

        # Log final metrics if available
        if info.metric:
            try:
                self._log_metric(info.metric, prefix="final_")
            except Exception as e:
                logger.warning(f"Could not log final metric: {e}")

        # Log model artifacts
        if info.artifacts:
            for artifact in info.artifacts:
                path = getattr(artifact, "path", None)
                if path and Path(path).exists():
                    try:
                        self.mlflow.log_artifact(str(path))
                    except Exception as e:
                        logger.warning(f"Could not log artifact {path}: {e}")

        # Log model code
        if info.predictor_source:
            try:
                # Save code to a file first, then log it
                code_path = Path("predictor_source.py")
                with open(code_path, "w") as f:
                    f.write(info.predictor_source)
                self.mlflow.log_artifact(str(code_path))
            except Exception as e:
                logger.warning(f"Could not log predictor source: {e}")

        if info.trainer_source:
            try:
                # Save code to a file first, then log it
                code_path = Path("trainer_source.py")
                with open(code_path, "w") as f:
                    f.write(info.trainer_source)
                self.mlflow.log_artifact(str(code_path))
            except Exception as e:
                logger.warning(f"Could not log trainer source: {e}")

        # Log metadata
        if info.metadata:
            for key, value in info.metadata.items():
                self.mlflow.set_tag(key, value)

        # Log error if any
        if info.error:
            self.mlflow.set_tag("error", str(info.error))

        self.mlflow.end_run()
        logger.info("Ended MLFlow parent run")

    def on_iteration_start(self, info: IterationStartInfo) -> None:
        """
        Start a new child run for this iteration if using nested runs.

        :param info: Information about the iteration start.
        """
        if self.mlflow is None or not self.mlflow.active_run():
            return

        if self.nested:
            self.mlflow.start_run(run_name=f"iteration-{info.iteration}", nested=True)

            # Log iteration information
            params = {"iteration": info.iteration}

            if info.target_metric:
                params["target_metric_name"] = info.target_metric.name
                params["target_metric_comparator"] = str(info.target_metric.comparator)

            if info.stopping_condition:
                params["max_iterations"] = info.stopping_condition.max_generations
                params["max_time"] = info.stopping_condition.max_time

            self.mlflow.log_params(params)
            logger.debug(f"Started MLFlow child run for iteration {info.iteration}")

    def on_iteration_end(self, info: IterationEndInfo) -> None:
        """
        Log metrics for this iteration.

        :param info: Information about the iteration end.
        """
        if self.mlflow is None or not self.mlflow.active_run():
            return

        # Log node performance if available
        if info.node.performance:
            self._log_metric(info.node.performance, step=info.iteration)

        # Log best performance so far
        if info.best_metric:
            self._log_metric(info.best_metric, prefix="best_", step=info.iteration)

        # Log execution time
        if info.node.execution_time:
            self.mlflow.log_metric("execution_time", info.node.execution_time, step=info.iteration)

        # Log elapsed and remaining time
        if info.elapsed_time:
            self.mlflow.log_metric("elapsed_time", info.elapsed_time, step=info.iteration)

        if info.remaining_time:
            self.mlflow.log_metric("remaining_time", info.remaining_time, step=info.iteration)

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

        if self.nested:
            self.mlflow.end_run()

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
