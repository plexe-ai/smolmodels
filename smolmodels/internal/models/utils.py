import json

from smolmodels.internal.models.entities.node import Node
from smolmodels.internal.models.execution.executor import Executor


def join_problem_statement(intent: str, input_schema: dict, output_schema: dict, constraints, directives) -> str:
    """Join the problem statement into a single string."""
    return (
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


def execute_node(node: Node, executor: Executor) -> None:
    """
    Execute the training code for the given node using the executor.
    """

    result = executor.run()
    node.execution_time = result.exec_time
    node.execution_stdout = result.term_out
    node.execution_stderr = result.exc_stack or None
    node.exception_was_raised = result.exc_type is not None
    node.exception = result.exc_stack or None
    node.model_artifacts = result.model_artifacts
    node.analysis = result.analysis
    node.performance = result.performance
