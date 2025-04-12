"""
Tools related to code generation, including solution planning, training code, 
and inference code generation.
"""

import logging
from typing import List, Dict
from smolagents import tool

from smolmodels.internal.common.provider import Provider
from smolmodels.internal.models.generation.planning import SolutionPlanGenerator
from smolmodels.internal.models.generation.training import TrainingCodeGenerator
from smolmodels.internal.models.generation.inference import InferenceCodeGenerator

logger = logging.getLogger(__name__)


@tool
def generate_solution_plan(task: str, metric_name: str, provider: str) -> str:
    """Generates a solution plan for the given task optimizing for the provided metric.

    Args:
        task: The task definition combining intent, input schema, and output schema
        metric_name: The name of the metric to optimize for
        provider: The string identifier representing the model that should be used, e.g. 'openai/gpt-4o'

    Returns:
        A solution plan as a structured string
    """
    plan_generator = SolutionPlanGenerator(Provider(provider))
    return plan_generator.generate_solution_plan(task, metric_name)


@tool
def generate_training_code(
    task: str, solution_plan: str, train_datasets: List[str], validation_datasets: List[str], provider: str
) -> str:
    """Generates training code based on the solution plan.

    Args:
        task: The task definition
        solution_plan: The solution plan to implement
        train_datasets: Keys of datasets to use for training
        validation_datasets: Keys of datasets to use for validation
        provider: The string identifier representing the model that should be used, e.g. 'openai/gpt-4o'

    Returns:
        Generated training code as a string
    """
    train_generator = TrainingCodeGenerator(Provider(provider))
    return train_generator.generate_training_code(task, solution_plan, train_datasets, validation_datasets)


@tool
def review_training_code(training_code: str, task: str, solution_plan: str, issue: str, provider: str) -> str:
    """Reviews training code and provides suggestions for improvement.

    Args:
        training_code: The training code to review
        task: The task definition
        solution_plan: The solution plan being implemented
        issue: Description of the issue to address
        provider: The string identifier representing the model that should be used, e.g. 'openai/gpt-4o'

    Returns:
        Review comments as a string
    """
    train_generator = TrainingCodeGenerator(Provider(provider))
    return train_generator.review_training_code(training_code, task, solution_plan, issue)


@tool
def fix_training_code(
    training_code: str,
    solution_plan: str,
    review: str,
    train_datasets: List[str],
    validation_datasets: List[str],
    issue: str,
    provider: str,
) -> str:
    """Fixes issues in the training code based on the review.

    Args:
        training_code: The training code to fix
        solution_plan: The solution plan being implemented
        review: Review comments about the code
        train_datasets: Keys of datasets to use for training
        validation_datasets: Keys of datasets to use for validation
        issue: Description of the issue to address
        provider: The string identifier representing the model that should be used, e.g. 'openai/gpt-4o'

    Returns:
        Fixed training code as a string
    """
    train_generator = TrainingCodeGenerator(Provider(provider))
    return train_generator.fix_training_code(
        training_code, solution_plan, review, train_datasets, validation_datasets, issue
    )


@tool
def generate_inference_code(
    input_schema: Dict[str, type], output_schema: Dict[str, type], training_code: str, provider: str
) -> str:
    """
    Generates inference code based on the training code.

    Args:
        input_schema: The input schema for the model, for example {"feat_1": int, "feat_2": str}
        output_schema: The output schema for the model, for example {"output": float}
        training_code: The training code that was used
        provider: The string identifier representing the model that should be used, e.g. 'openai/gpt-4o'

    Returns:
        Generated inference code as a string
    """
    from smolmodels.internal.common.utils.pydantic_utils import map_to_basemodel

    try:
        # Convert dict schemas to Type[BaseModel]
        input_model = map_to_basemodel("InputSchema", input_schema)
        output_model = map_to_basemodel("OutputSchema", output_schema)

        infer_generator = InferenceCodeGenerator(Provider(provider))
        return infer_generator.generate_inference_code(input_model, output_model, training_code)
    except Exception as e:
        raise ValueError(f"Failed to generate inference code: {str(e)}") from e


@tool
def review_inference_code(
    inference_code: str,
    input_schema: Dict[str, type],
    output_schema: Dict[str, type],
    training_code: str,
    problems: str,
    provider: str,
) -> str:
    """
    Reviews inference code and provides suggestions for improvement.

    Args:
        inference_code: The inference code to review
        input_schema: The input schema for the model
        output_schema: The output schema for the model
        training_code: The training code that was used
        problems: Description of the problems to address
        provider: The string identifier representing the model that should be used, e.g. 'openai/gpt-4o'

    Returns:
        Review comments as a string
    """
    from smolmodels.internal.common.utils.pydantic_utils import map_to_basemodel

    # Convert dict schemas to Type[BaseModel]
    input_model = map_to_basemodel("InputSchema", input_schema)
    output_model = map_to_basemodel("OutputSchema", output_schema)

    infer_generator = InferenceCodeGenerator(Provider(provider))
    return infer_generator.review_inference_code(inference_code, input_model, output_model, training_code, problems)


@tool
def fix_inference_code(inference_code: str, review: str, problems: str, provider: str) -> str:
    """
    Fixes issues in the inference code based on the review.

    Args:
        inference_code: The inference code to fix
        review: Review comments about the code
        problems: Description of the problems to address
        provider: The string identifier representing the model that should be used, e.g. 'openai/gpt-4o'

    Returns:
        Fixed inference code as a string
    """
    infer_generator = InferenceCodeGenerator(Provider(provider))
    return infer_generator.fix_inference_code(inference_code, review, problems)
