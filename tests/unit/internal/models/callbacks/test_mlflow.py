"""
Unit tests for the MLFlowCallback class.

These tests validate the functionality of the MLFlowCallback, which is responsible
for logging model building metrics, parameters, and artifacts to MLFlow.

The tests cover:
- Initialization and configuration of MLFlow tracking
- Lifecycle callbacks (build start/end, iteration start/end)
- Metric logging, parameter logging, and artifact logging
- Error handling and optional dependency management
- Nested run support

The tests use mocking to avoid actual MLFlow server dependencies.
"""

from unittest.mock import MagicMock, patch, call

import pytest
from pydantic import BaseModel

from smolmodels.callbacks import BuildStartInfo, BuildEndInfo, IterationStartInfo, IterationEndInfo
from smolmodels.internal.common.utils.model_state import ModelState
from smolmodels.internal.models.callbacks.mlflow import MLFlowCallback
from smolmodels.internal.models.entities.artifact import Artifact
from smolmodels.internal.models.entities.metric import Metric, MetricComparator, ComparisonMethod
from smolmodels.internal.models.entities.node import Node
from smolmodels.internal.models.entities.stopping_condition import StoppingCondition


class TestMLFlowCallback:
    """Test suite for the MLFlowCallback class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create common test objects
        self.metric = Metric(
            name="accuracy", value=0.95, comparator=MetricComparator(ComparisonMethod.HIGHER_IS_BETTER)
        )
        self.node = Node(
            solution_plan="Train a random forest model",
            performance=self.metric,
            execution_time=10.5,
            model_artifacts=["/path/to/artifact.pkl"],
        )

        # Mock input/output schemas
        class InputSchema(BaseModel):
            feature1: float
            feature2: str

        class OutputSchema(BaseModel):
            prediction: float

        self.input_schema = InputSchema
        self.output_schema = OutputSchema

        # Create info objects for callbacks
        self.build_start_info = BuildStartInfo(
            intent="Predict house prices",
            identifier="model-12345",
            provider="openai/gpt-4o-mini",
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            run_timeout=300,
            max_iterations=10,
            timeout=3600,
        )

        self.build_end_info = BuildEndInfo(
            intent="Predict house prices",
            identifier="model-12345",
            provider="openai/gpt-4o-mini",
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            run_timeout=300,
            max_iterations=10,
            timeout=3600,
            state=ModelState.READY,
            metric=self.metric,
            artifacts=[Artifact(name="model.pkl", path="/path/to/model.pkl")],
            trainer_source="def train(): pass",
            predictor_source="def predict(x): return {'prediction': 0.5}",
            metadata={"framework": "scikit-learn"},
        )

        self.iteration_start_info = IterationStartInfo(
            iteration=1,
            total_iterations=10,
            target_metric=self.metric,
            stopping_condition=StoppingCondition(10, 3600, None),
            elapsed_time=60.0,
        )

        self.iteration_end_info = IterationEndInfo(
            iteration=1, node=self.node, best_metric=self.metric, elapsed_time=65.0, remaining_time=3535.0
        )

        # Create mock MLFlow
        self.mock_mlflow = MagicMock()
        self.mock_mlflow.active_run.return_value = MagicMock()
        self.mock_mlflow.active_run.return_value.info.run_id = "test-run-id"
        self.mock_mlflow.get_experiment_by_name.return_value = None
        self.mock_mlflow.create_experiment.return_value = "test-experiment-id"

        # Default callback parameters for tests
        self.default_tracking_uri = "http://localhost:5000"
        self.default_experiment_name = "test-experiment"

    def test_import_mlflow_success(self):
        """Test _import_mlflow method when MLFlow is available."""
        # Create a real MLFlowCallback instance
        callback = MLFlowCallback(tracking_uri=self.default_tracking_uri, experiment_name=self.default_experiment_name)

        # Test _import_mlflow directly with mocked import
        with patch("builtins.__import__", return_value=self.mock_mlflow):
            result = callback._import_mlflow()
            assert result is self.mock_mlflow

    def test_import_mlflow_failure(self):
        """Test _import_mlflow method when MLFlow is not available."""
        # Create a real MLFlowCallback instance
        callback = MLFlowCallback(tracking_uri=self.default_tracking_uri, experiment_name=self.default_experiment_name)

        # Test _import_mlflow directly with mocked import that raises ImportError
        with patch("builtins.__import__", side_effect=ImportError):
            with patch("smolmodels.internal.models.callbacks.mlflow.logger") as mock_logger:
                result = callback._import_mlflow()
                assert result is None
                mock_logger.warning.assert_called_once()

    def test_init_with_mlflow_available(self):
        """Test initialization with MLFlow available."""
        with patch.object(MLFlowCallback, "_import_mlflow", return_value=self.mock_mlflow):
            callback = MLFlowCallback(
                tracking_uri=self.default_tracking_uri, experiment_name=self.default_experiment_name
            )

            # Verify MLFlow was configured correctly
            assert callback.mlflow is self.mock_mlflow
            self.mock_mlflow.set_tracking_uri.assert_called_once_with(self.default_tracking_uri)
            self.mock_mlflow.get_experiment_by_name.assert_called_once_with(self.default_experiment_name)
            self.mock_mlflow.create_experiment.assert_called_once_with(self.default_experiment_name)
            assert callback.experiment_id == "test-experiment-id"

    def test_init_with_existing_experiment(self):
        """Test initialization with an existing MLFlow experiment."""
        mock_mlflow = MagicMock()
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "existing-experiment-id"
        mock_mlflow.get_experiment_by_name.return_value = mock_experiment

        with patch.object(MLFlowCallback, "_import_mlflow", return_value=mock_mlflow):
            callback = MLFlowCallback(tracking_uri=self.default_tracking_uri, experiment_name="existing-experiment")

            # Verify experiment is reused, not created
            mock_mlflow.get_experiment_by_name.assert_called_once_with("existing-experiment")
            mock_mlflow.create_experiment.assert_not_called()
            assert callback.experiment_id == "existing-experiment-id"

    def test_init_with_mlflow_unavailable(self):
        """Test initialization with MLFlow unavailable."""
        with patch.object(MLFlowCallback, "_import_mlflow", return_value=None):
            with patch("smolmodels.internal.models.callbacks.mlflow.logger"):
                callback = MLFlowCallback(
                    tracking_uri=self.default_tracking_uri, experiment_name=self.default_experiment_name
                )

                # No mlflow setup should occur
                assert callback.mlflow is None
                # _setup_mlflow is called but it checks self.mlflow, which is None
                assert callback.experiment_id is None

    def test_on_build_start(self):
        """Test on_build_start callback."""
        with patch.object(MLFlowCallback, "_import_mlflow", return_value=self.mock_mlflow):
            callback = MLFlowCallback(
                tracking_uri=self.default_tracking_uri, experiment_name=self.default_experiment_name
            )
            callback.on_build_start(self.build_start_info)

            # Verify MLFlow start_run was called
            self.mock_mlflow.start_run.assert_called_once()

            # Verify parameters were logged
            self.mock_mlflow.log_params.assert_called_once()
            params_arg = self.mock_mlflow.log_params.call_args[0][0]
            assert params_arg["intent"] == "Predict house prices"
            assert params_arg["run_timeout"] == 300
            assert params_arg["max_iterations"] == 10
            assert params_arg["timeout"] == 3600

            # Verify we get the parent run ID
            assert callback.parent_run_id == "test-run-id"

    def test_on_build_start_schema_exceptions(self):
        """Test on_build_start with various schema-related exceptions."""
        with patch.object(MLFlowCallback, "_import_mlflow", return_value=self.mock_mlflow):
            callback = MLFlowCallback(
                tracking_uri=self.default_tracking_uri, experiment_name=self.default_experiment_name
            )

            # Test both input and output schema exceptions
            with patch.object(self.input_schema, "model_json_schema", side_effect=Exception("Input schema error")):
                with patch.object(
                    self.output_schema, "model_json_schema", side_effect=Exception("Output schema error")
                ):
                    with patch("smolmodels.internal.models.callbacks.mlflow.logger") as mock_logger:
                        callback.on_build_start(self.build_start_info)

                        # Verify both warnings were logged
                        assert mock_logger.warning.call_count == 2

    def test_on_build_start_with_existing_parent_run(self):
        """Test on_build_start callback with existing parent run ID."""
        with patch.object(MLFlowCallback, "_import_mlflow", return_value=self.mock_mlflow):
            callback = MLFlowCallback(
                tracking_uri=self.default_tracking_uri,
                experiment_name=self.default_experiment_name,
                parent_run_id="existing-run-id",
            )
            callback.on_build_start(self.build_start_info)

            # Verify MLFlow start_run was called with the existing run ID
            self.mock_mlflow.start_run.assert_called_once_with(run_id="existing-run-id", nested=True)

    def test_on_build_end(self):
        """Test on_build_end callback."""
        # Setup file existence checks
        with patch("pathlib.Path.exists", return_value=True):
            with patch.object(MLFlowCallback, "_import_mlflow", return_value=self.mock_mlflow):
                with patch("builtins.open", MagicMock()):
                    callback = MLFlowCallback(
                        tracking_uri=self.default_tracking_uri, experiment_name=self.default_experiment_name
                    )
                    callback.on_build_end(self.build_end_info)

                    # Verify metrics, artifacts, and tags were logged
                    self.mock_mlflow.set_tag.assert_any_call("model_state", "ready")
                    self.mock_mlflow.log_metric.assert_called_once()
                    self.mock_mlflow.log_artifact.assert_called()
                    self.mock_mlflow.end_run.assert_called_once()

    def test_on_build_end_with_nonexistent_artifacts(self):
        """Test on_build_end with artifacts that don't exist."""
        # Setup file existence checks to return False
        with patch("pathlib.Path.exists", return_value=False):
            with patch.object(MLFlowCallback, "_import_mlflow", return_value=self.mock_mlflow):
                with patch("builtins.open", MagicMock()):
                    callback = MLFlowCallback(
                        tracking_uri=self.default_tracking_uri, experiment_name=self.default_experiment_name
                    )
                    callback.on_build_end(self.build_end_info)

                    # Artifacts shouldn't be logged if they don't exist
                    for call_args in self.mock_mlflow.log_artifact.call_args_list:
                        assert "/path/to/model.pkl" not in call_args[0]

    def test_on_build_end_with_error(self):
        """Test on_build_end callback with error."""
        error_info = BuildEndInfo(
            intent="Predict house prices",
            identifier="model-12345",
            provider="openai/gpt-4o-mini",
            state=ModelState.ERROR,
            error=ValueError("Failed to build model"),
        )

        with patch.object(MLFlowCallback, "_import_mlflow", return_value=self.mock_mlflow):
            callback = MLFlowCallback(
                tracking_uri=self.default_tracking_uri, experiment_name=self.default_experiment_name
            )
            callback.on_build_end(error_info)

            # Verify error was logged
            self.mock_mlflow.set_tag.assert_any_call("model_state", "error")
            self.mock_mlflow.set_tag.assert_any_call("error", "Failed to build model")
            self.mock_mlflow.end_run.assert_called_once()

    def test_on_iteration_start_with_nested_runs(self):
        """Test on_iteration_start callback with nested runs enabled."""
        with patch.object(MLFlowCallback, "_import_mlflow", return_value=self.mock_mlflow):
            callback = MLFlowCallback(
                tracking_uri=self.default_tracking_uri, experiment_name=self.default_experiment_name, nested=True
            )
            callback.on_iteration_start(self.iteration_start_info)

            # Verify nested run was started
            self.mock_mlflow.start_run.assert_called_once_with(
                run_name=f"iteration-{self.iteration_start_info.iteration}", nested=True
            )

            # Verify parameters were logged
            self.mock_mlflow.log_params.assert_called_once()

    def test_on_iteration_start_without_active_run(self):
        """Test on_iteration_start when there's no active run."""
        mock_mlflow = MagicMock()
        mock_mlflow.active_run.return_value = None

        with patch.object(MLFlowCallback, "_import_mlflow", return_value=mock_mlflow):
            callback = MLFlowCallback(
                tracking_uri=self.default_tracking_uri, experiment_name=self.default_experiment_name, nested=True
            )
            # This should return immediately and not try to use MLFlow
            callback.on_iteration_start(self.iteration_start_info)

            # Verify no MLFlow API calls were made
            mock_mlflow.start_run.assert_not_called()
            mock_mlflow.log_params.assert_not_called()

    def test_on_iteration_start_without_nested_runs(self):
        """Test on_iteration_start callback with nested runs disabled."""
        with patch.object(MLFlowCallback, "_import_mlflow", return_value=self.mock_mlflow):
            callback = MLFlowCallback(
                tracking_uri=self.default_tracking_uri, experiment_name=self.default_experiment_name, nested=False
            )
            callback.on_iteration_start(self.iteration_start_info)

            # Verify nested run was not started
            self.mock_mlflow.start_run.assert_not_called()

    def test_on_iteration_end(self):
        """Test on_iteration_end callback."""
        with patch("pathlib.Path.exists", return_value=True):
            with patch.object(MLFlowCallback, "_import_mlflow", return_value=self.mock_mlflow):
                # Test with nested=True
                callback = MLFlowCallback(
                    tracking_uri=self.default_tracking_uri, experiment_name=self.default_experiment_name, nested=True
                )
                callback.on_iteration_end(self.iteration_end_info)

                # Verify metrics were logged
                assert self.mock_mlflow.log_metric.call_count >= 3

                # Verify execution time and elapsed time were logged
                calls = self.mock_mlflow.log_metric.call_args_list
                assert call("execution_time", 10.5, step=1) in calls
                assert call("elapsed_time", 65.0, step=1) in calls
                assert call("remaining_time", 3535.0, step=1) in calls

                # Verify artifacts were logged
                self.mock_mlflow.log_artifact.assert_called_with("/path/to/artifact.pkl")

                # Verify run was ended
                self.mock_mlflow.end_run.assert_called_once()

                # Reset mocks for the next test
                self.mock_mlflow.reset_mock()

                # Test without nested runs to cover different code paths
                callback = MLFlowCallback(
                    tracking_uri=self.default_tracking_uri, experiment_name=self.default_experiment_name, nested=False
                )
                callback.on_iteration_end(self.iteration_end_info)

                # When not using nested runs, end_run should not be called
                self.mock_mlflow.end_run.assert_not_called()

    def test_on_iteration_end_with_nonexistent_artifacts(self):
        """Test on_iteration_end with artifacts that don't exist."""
        with patch("pathlib.Path.exists", return_value=False):
            with patch.object(MLFlowCallback, "_import_mlflow", return_value=self.mock_mlflow):
                callback = MLFlowCallback(
                    tracking_uri=self.default_tracking_uri, experiment_name=self.default_experiment_name, nested=True
                )
                callback.on_iteration_end(self.iteration_end_info)

                # Artifacts shouldn't be logged if they don't exist
                for call_args in self.mock_mlflow.log_artifact.call_args_list:
                    assert "/path/to/artifact.pkl" not in call_args[0]

    def test_on_iteration_end_with_exception(self):
        """Test on_iteration_end callback with exception."""
        # Create a node with an exception
        node_with_error = Node(
            solution_plan="Train a random forest model",
            exception_was_raised=True,
            exception=ValueError("Training failed"),
        )

        info = IterationEndInfo(iteration=1, node=node_with_error, elapsed_time=65.0)

        with patch.object(MLFlowCallback, "_import_mlflow", return_value=self.mock_mlflow):
            callback = MLFlowCallback(
                tracking_uri=self.default_tracking_uri, experiment_name=self.default_experiment_name, nested=True
            )
            callback.on_iteration_end(info)

            # Verify exception was logged
            self.mock_mlflow.set_tag.assert_called_with("iteration_1_error", "Training failed")

    def test_log_metric_success(self):
        """Test _log_metric helper function with valid metric."""
        with patch.object(MLFlowCallback, "_import_mlflow", return_value=self.mock_mlflow):
            callback = MLFlowCallback(
                tracking_uri=self.default_tracking_uri, experiment_name=self.default_experiment_name
            )
            callback._log_metric(self.metric, prefix="test_", step=5)

            # Verify metric was logged with correct name, value, and step
            self.mock_mlflow.log_metric.assert_called_once_with("test_accuracy", 0.95, step=5)

    def test_log_metric_with_non_float_value(self):
        """Test _log_metric with non-float metric value."""
        # Create a metric with non-float value
        metric = Metric(
            name="classification", value="good", comparator=MetricComparator(ComparisonMethod.HIGHER_IS_BETTER)
        )

        with patch.object(MLFlowCallback, "_import_mlflow", return_value=self.mock_mlflow):
            with patch("smolmodels.internal.models.callbacks.mlflow.logger") as mock_logger:
                callback = MLFlowCallback(
                    tracking_uri=self.default_tracking_uri, experiment_name=self.default_experiment_name
                )
                callback._log_metric(metric)

                # Verify metric was logged as a tag instead
                self.mock_mlflow.log_metric.assert_not_called()
                self.mock_mlflow.set_tag.assert_called_once_with("classification", "good")
                mock_logger.warning.assert_called_once()

    def test_without_active_run(self):
        """Test callbacks without an active run."""
        mock_mlflow = MagicMock()
        mock_mlflow.active_run.return_value = None

        with patch.object(MLFlowCallback, "_import_mlflow", return_value=mock_mlflow):
            callback = MLFlowCallback(
                tracking_uri=self.default_tracking_uri, experiment_name=self.default_experiment_name
            )
            # These calls should not raise exceptions, just return quietly
            callback.on_build_end(self.build_end_info)
            callback.on_iteration_start(self.iteration_start_info)
            callback.on_iteration_end(self.iteration_end_info)

            # Verify no MLFlow logging happened
            mock_mlflow.log_metric.assert_not_called()
            mock_mlflow.log_params.assert_not_called()
            mock_mlflow.log_artifact.assert_not_called()

    def test_schema_logging_errors(self):
        """Test error handling during schema logging."""
        # Create patched schema that raises an exception
        with patch.object(self.input_schema, "model_json_schema", side_effect=Exception("Schema error")):
            with patch.object(MLFlowCallback, "_import_mlflow", return_value=self.mock_mlflow):
                with patch("smolmodels.internal.models.callbacks.mlflow.logger") as mock_logger:
                    callback = MLFlowCallback(
                        tracking_uri=self.default_tracking_uri, experiment_name=self.default_experiment_name
                    )
                    callback.on_build_start(self.build_start_info)

                    # Verify warning was logged
                    mock_logger.warning.assert_called_once()

                    # Verify other parameters were still logged
                    self.mock_mlflow.log_params.assert_called_once()

    def test_artifact_logging_errors(self):
        """Test error handling during artifact logging."""
        with patch("pathlib.Path.exists", return_value=True):
            with patch.object(MLFlowCallback, "_import_mlflow", return_value=self.mock_mlflow):
                with patch.object(self.mock_mlflow, "log_artifact", side_effect=Exception("Artifact error")):
                    with patch("smolmodels.internal.models.callbacks.mlflow.logger") as mock_logger:
                        callback = MLFlowCallback(
                            tracking_uri=self.default_tracking_uri, experiment_name=self.default_experiment_name
                        )
                        callback.on_build_end(self.build_end_info)

                        # Verify warning was logged
                        mock_logger.warning.assert_called()

                        # Verify run was still ended
                        self.mock_mlflow.end_run.assert_called_once()

    def test_no_mlflow(self):
        """Test that callbacks do nothing when MLFlow is not available."""
        with patch.object(MLFlowCallback, "_import_mlflow", return_value=None):
            callback = MLFlowCallback(
                tracking_uri=self.default_tracking_uri, experiment_name=self.default_experiment_name
            )

            # These should not raise errors
            callback.on_build_start(self.build_start_info)
            callback.on_build_end(self.build_end_info)
            callback.on_iteration_start(self.iteration_start_info)
            callback.on_iteration_end(self.iteration_end_info)
            callback._log_metric(self.metric)

    def test_code_logging(self):
        """Test code artifact logging."""
        with patch.object(MLFlowCallback, "_import_mlflow", return_value=self.mock_mlflow):
            # Mock the file open calls
            mock_open = MagicMock()
            with patch("builtins.open", mock_open):
                with patch("pathlib.Path", MagicMock()):
                    callback = MLFlowCallback(
                        tracking_uri=self.default_tracking_uri, experiment_name=self.default_experiment_name
                    )
                    callback.on_build_end(self.build_end_info)

                    # Verify predictor and trainer code was written to files
                    mock_open.assert_called()
                    assert self.mock_mlflow.log_artifact.call_count >= 2

    def test_code_logging_exceptions(self):
        """Test error handling during code logging."""
        with patch.object(MLFlowCallback, "_import_mlflow", return_value=self.mock_mlflow):
            # First, test predictor code logging error
            with patch("builtins.open", side_effect=[Exception("File error"), MagicMock()]):
                with patch("smolmodels.internal.models.callbacks.mlflow.logger") as mock_logger:
                    callback = MLFlowCallback(
                        tracking_uri=self.default_tracking_uri, experiment_name=self.default_experiment_name
                    )
                    callback.on_build_end(self.build_end_info)

                    # Verify warning was logged
                    mock_logger.warning.assert_called_once()

            # Reset mocks
            self.mock_mlflow.reset_mock()

            # Next, test trainer code logging error
            with patch("builtins.open", side_effect=[MagicMock(), Exception("File error")]):
                with patch("smolmodels.internal.models.callbacks.mlflow.logger") as mock_logger:
                    callback = MLFlowCallback(
                        tracking_uri=self.default_tracking_uri, experiment_name=self.default_experiment_name
                    )
                    callback.on_build_end(self.build_end_info)

                    # Verify warning was logged
                    mock_logger.warning.assert_called_once()


if __name__ == "__main__":
    pytest.main()
