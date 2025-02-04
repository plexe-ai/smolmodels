import json
import logging
import textwrap

from smolmodels.internal.models.entities.metric import Metric
from smolmodels.internal.models.entities.node import Node
from smolmodels.internal.models.execution.executor import Executor

logger = logging.getLogger(__name__)


def join_task_statement(intent: str, input_schema: dict, output_schema: dict, constraints, directives) -> str:
    """Join the problem statement into a single string."""
    problem_statement: str = (
        "# Problem Statement"
        "\n\n"
        f"{intent}"
        "\n\n"
        "# Input Schema"
        "\n\n"
        f"{json.dumps(input_schema, indent=4, default=str)}"
        "\n\n"
        "# Output Schema"
        "\n\n"
        f"{json.dumps(output_schema, indent=4, default=str)}"
        "\n\n"
        "# Constraints"
        "\n\n"
        f"{json.dumps(constraints, indent=4, default=str)}"
        "\n\n"
        "# Directives"
        "\n\n"
        f"{json.dumps(directives, indent=4, default=str)}"
    )
    logger.debug(f"Joined user inputs into problem statement: {textwrap.shorten(problem_statement, 40)}")
    return problem_statement


def execute_node(node: Node, executor: Executor, metric_to_optimise: Metric) -> None:
    """
    Execute the training code for the given node using the executor.
    """
    logger.debug(f"Executing node {node} using executor {executor}")
    result = executor.run()
    logger.debug(f"Execution result: {result}")
    node.execution_time = result.exec_time
    node.execution_stdout = result.term_out
    node.exception_was_raised = result.exception is not None
    node.exception = result.exception or None
    node.model_artifacts = result.model_artifacts
    node.performance = Metric(metric_to_optimise.name, result.performance, metric_to_optimise.comparator)
    logger.debug(f"Unpacked execution results into node: {node}")
