"""
Tools related to code execution, including running training code in isolated environments.

These tools automatically handle model artifact registration through the ArtifactRegistry,
ensuring that artifacts generated during the execution can be retrieved later in the pipeline.
"""

import logging
import uuid
from typing import Dict, List

from smolagents import tool

from smolmodels.internal.models.entities.metric import Metric, MetricComparator, ComparisonMethod
from smolmodels.internal.models.entities.node import Node
from smolmodels.internal.models.execution.process_executor import ProcessExecutor
from smolmodels.internal.common.registries.datasets import DatasetRegistry
from smolmodels.internal.common.registries.artifacts import ArtifactRegistry

logger = logging.getLogger(__name__)


@tool
def execute_training_code(
    node_id: str,
    code: str,
    working_dir: str,
    dataset_names: List[str],
    timeout: int,
    metric_to_optimize: Dict,
) -> Dict:
    """Executes training code in an isolated environment.

    Args:
        node_id: Unique identifier for this execution
        code: The code to execute
        working_dir: Directory to use for execution
        dataset_names: List of dataset names to retrieve from the registry
        timeout: Maximum execution time in seconds
        metric_to_optimize: The metric to optimize for

    Returns:
        A dictionary containing execution results with model artifacts and their registry names
    """
    execution_id = f"{node_id}-{uuid.uuid4()}"
    try:
        # Get actual datasets from registry
        datasets = DatasetRegistry().get_multiple(dataset_names)

        # Convert metric_to_optimize back to a Metric object
        # Handle missing keys and set reasonable defaults
        metric_name = metric_to_optimize.get("name", "unknown")

        # Set a default value to negative infinity for metrics to maximize
        # or positive infinity for metrics to minimize
        import math

        default_value = -math.inf
        metric_value = metric_to_optimize.get("value", default_value)

        # Handle the comparison_method properly, as it may come as a string
        comparison_method_value = metric_to_optimize.get("comparison_method", "ComparisonMethod.HIGHER_IS_BETTER")

        # Convert string to enum if needed
        if isinstance(comparison_method_value, str):
            if "HIGHER_IS_BETTER" in comparison_method_value:
                comparison_method = ComparisonMethod.HIGHER_IS_BETTER
            elif "LOWER_IS_BETTER" in comparison_method_value:
                comparison_method = ComparisonMethod.LOWER_IS_BETTER
                # For lower is better metrics, default should be positive infinity
                if "value" not in metric_to_optimize:
                    metric_value = math.inf
            elif "TARGET_IS_BETTER" in comparison_method_value:
                comparison_method = ComparisonMethod.TARGET_IS_BETTER
            else:
                # Default to higher is better if unknown
                comparison_method = ComparisonMethod.HIGHER_IS_BETTER
        else:
            # Assume it's already a ComparisonMethod enum
            comparison_method = comparison_method_value

        metric = Metric(
            name=metric_name,
            value=metric_value,
            comparator=MetricComparator(comparison_method=comparison_method),
        )

        # Create a node to store execution results
        node = Node(solution_plan="")  # We only need this for execute_node

        # Import here to avoid circular imports
        from smolmodels.config import config

        executor = ProcessExecutor(
            execution_id=execution_id,
            code=code,
            working_dir=working_dir,
            datasets=datasets,
            timeout=timeout,
            code_execution_file_name=config.execution.runfile_name,
        )

        logger.debug(f"Executing node {node} using executor {executor}")
        result = executor.run()
        logger.debug(f"Execution result: {result}")
        node.execution_time = result.exec_time
        node.execution_stdout = result.term_out
        node.exception_was_raised = result.exception is not None
        node.exception = result.exception or None
        node.model_artifacts = result.model_artifacts

        # Handle the performance metric properly
        is_worst = True
        performance_value = None

        if isinstance(result.performance, (int, float)):
            performance_value = result.performance
            is_worst = False

        # Create a metric object with proper handling of None or invalid values
        node.performance = Metric(
            name=metric.name,
            value=performance_value,
            comparator=metric.comparator,
            is_worst=is_worst,
        )
        logger.debug(f"Unpacked execution results into node: {node}")

        # Get model artifacts from node
        artifact_paths = node.model_artifacts if node.model_artifacts else []

        # Register artifacts directly in the ArtifactRegistry
        artifact_names = []
        if artifact_paths:
            try:
                # Use ArtifactRegistry to register artifacts
                artifact_registry = ArtifactRegistry()
                artifact_names = artifact_registry.register_batch([str(path) for path in artifact_paths])
                logger.info(f"Registered {len(artifact_names)} artifacts with names: {artifact_names}")
            except Exception as e:
                logger.error(f"Error registering artifacts: {str(e)}")
                # Continue with empty names list

        # Return results
        return {
            "success": not node.exception_was_raised,
            "performance": (
                {
                    "name": node.performance.name if node.performance else None,
                    "value": node.performance.value if node.performance else None,
                    "comparison_method": (
                        str(node.performance.comparator.comparison_method) if node.performance else None
                    ),
                }
                if node.performance
                else None
            ),
            "exception": str(node.exception) if node.exception else None,
            "model_artifacts": [str(artifact) for artifact in artifact_paths],
            "model_artifact_names": artifact_names,
        }
    except Exception as e:
        logger.error(f"Error executing training code: {str(e)}")
        return {
            "success": False,
            "performance": None,
            "exception": str(e),
            "model_artifacts": [],
            "model_artifact_names": [],
        }
