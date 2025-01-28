# smolmodels/internal/models/generation/planning.py

"""
This module provides functions and classes for generating and planning solutions for machine learning problems.
"""

import json
import logging
from typing import List, Dict

import pandas as pd
from pydantic import BaseModel

from smolmodels.config import config
from smolmodels.internal.common.providers.openai import OpenAIProvider
from smolmodels.internal.models.entities.metric import Metric, MetricComparator, ComparisonMethod
from smolmodels.internal.models.entities.stopping_condition import StoppingCondition

logger = logging.getLogger(__name__)

client = OpenAIProvider()


def select_metric_to_optimise(problem_statement: str, dataset: pd.DataFrame) -> Metric:
    """
    Selects the metric to optimise for the given problem statement and dataset.

    :param problem_statement: definition of the problem
    :param dataset: data used for training and evaluation
    :return: the metric to optimise
    """

    class MetricResponse(BaseModel):
        name: str
        comparison_method: ComparisonMethod
        comparison_target: float = None

    response: MetricResponse = MetricResponse(
        **json.loads(
            client.query(
                system_message=config.code_generation.prompt_planning_select_metric.safe_substitute(),
                user_message=config.code_generation.prompt_planning_select_metric.safe_substitute(
                    problem_statement=problem_statement,
                ),
                response_format=MetricResponse,
            )
        )
    )

    try:
        return Metric(
            name=response.name,
            value=float("inf") if response.comparison_method == ComparisonMethod.LOWER_IS_BETTER else -float("inf"),
            comparator=MetricComparator(response.comparison_method, response.comparison_target),
        )
    except Exception as e:
        raise ValueError(f"Could not determine optimization metric from problem statement: {response}") from e


def select_stopping_condition(
    problem_statement: str, metric: Metric, max_iterations: int, max_time: int
) -> StoppingCondition:
    """
    Selects the stopping condition for the given problem statement and dataset.

    :param problem_statement: definition of the problem
    :param metric: the metric to optimise
    :param max_iterations: the maximum number of iterations
    :param max_time: the maximum time allowed
    :return: the stopping condition
    """

    class StoppingConditionResponse(BaseModel):
        max_generations: int
        max_time: int
        metric_threshold: float

    response: StoppingConditionResponse = StoppingConditionResponse(
        **json.loads(
            client.query(
                system_message=config.code_generation.prompt_planning_select_stop_condition.safe_substitute(),
                user_message=config.code_generation.prompt_planning_select_stop_condition.safe_substitute(
                    problem_statement=problem_statement,
                    metric=metric.name,
                ),
                response_format=StoppingConditionResponse,
            )
        )
    )

    try:
        return StoppingCondition(
            max_generations=min(response.max_generations, max_iterations),
            max_time=min(response.max_time, max_time),
            metric=Metric(metric.name, response.metric_threshold, metric.comparator),
        )
    except Exception as e:
        raise ValueError(f"Could not determine stopping condition from problem statement: {response}") from e


def generate_solution_plan(problem_statement: str, metric_to_optimise: str, context: str = None) -> str:
    """
    Generates a solution plan for the given problem statement.

    :param problem_statement: definition of the problem
    :param metric_to_optimise: the metric to optimise
    :param context: additional context or memory for the solution
    :return: the generated solution plan
    """
    return client.query(
        system_message=config.code_generation.prompt_planning_base.safe_substitute(),
        user_message=config.code_generation.prompt_planning_generate_plan.safe_substitute(
            problem_statement=problem_statement,
            metric_to_optimise=metric_to_optimise,
            context=context,
        ),
    )


class SolutionPlanGenerator:
    """
    A class to generate solution plans for given problem statements.
    """

    def __init__(self):
        """
        Initializes the SolutionPlanGenerator with an empty context.
        """
        self.context: List[Dict[str, str]] = []

    def generate_solution_plan(self, problem_statement: str) -> str:
        """
        Generates a solution plan for the given problem statement and updates the context.

        :param problem_statement: definition of the problem
        :return: the generated solution plan
        """
        solution = generate_solution_plan(problem_statement, str(self.context))
        self.context.append(
            {
                "problem_statement": problem_statement,
                "plan": solution,
            }
        )
        return solution
