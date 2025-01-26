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
                user_message=f"Problem statement:\n{problem_statement}\n\nDataset Schema:\n{dataset}",
                response_format=MetricResponse,
            )
        )
    )
    return Metric(
        name=response.name,
        value=float("inf") if response.comparison_method == ComparisonMethod.LOWER_IS_BETTER else -float("inf"),
        comparator=MetricComparator(response.comparison_method, response.comparison_target),
    )


def select_stopping_condition(problem_statement: str, dataset: pd.DataFrame, metric: Metric) -> StoppingCondition:
    """
    Selects the stopping condition for the given problem statement and dataset.

    :param problem_statement: definition of the problem
    :param dataset: data used for training and evaluation
    :param metric: the metric to optimise
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
                user_message=f"Problem:\n{problem_statement}\n\nDataset:\n{dataset}\n\nMetric:\n{metric}",
                response_format=StoppingConditionResponse,
            )
        )
    )

    return StoppingCondition(
        max_generations=response.max_generations,
        max_time=response.max_time,
        metric=Metric(metric.name, response.metric_threshold, metric.comparator),
    )


def generate_solution_plan(problem_statement: str, context: str = None) -> str:
    """
    Generates a solution plan for the given problem statement.

    :param problem_statement: definition of the problem
    :param context: additional context or memory for the solution
    :return: the generated solution plan
    """
    return client.query(
        system_message=config.code_generation.prompt_planning_base.safe_substitute(),
        user_message=config.code_generation.prompt_planning_generate_plan.safe_substitute(
            problem_statement=problem_statement,
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
