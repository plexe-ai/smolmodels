import json
import logging
import textwrap
from typing import Type

from pydantic import BaseModel

logger = logging.getLogger(__name__)


def join_task_statement(intent: str, input_schema: Type[BaseModel], output_schema: Type[BaseModel]) -> str:
    """Join the problem statement into a single string."""
    problem_statement: str = (
        "# Problem Statement"
        "\n\n"
        f"{intent}"
        "\n\n"
        "# Input Schema"
        "\n\n"
        f"{json.dumps(input_schema.model_fields, indent=4, default=str)}"
        "\n\n"
        "# Output Schema"
        "\n\n"
        f"{json.dumps(output_schema.model_fields, indent=4, default=str)}"
    )
    logger.debug(f"Joined user inputs into problem statement: {textwrap.shorten(problem_statement, 40)}")
    return problem_statement
