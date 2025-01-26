# smolmodels/internal/models/generation/planning.py

"""
This module provides functions and classes for generating and planning solutions for machine learning problems.
"""

import logging
from typing import List, Dict

import openai
from pydantic import BaseModel

from smolmodels.config import config
from smolmodels.internal.models.entities.metric import Metric, MetricComparator, ComparisonMethod
from smolmodels.internal.models.entities.stopping_condition import StoppingCondition

logger = logging.getLogger(__name__)

client = openai.OpenAI()


def select_metric_to_optimise(problem_statement: str, dataset: str) -> Metric:
    """
    Selects the metric to optimise for the given problem statement and dataset.

    :param problem_statement: definition of the problem
    :param dataset: data used for training and evaluation
    :return: the metric to optimise
    """

    class ResponseFormat(BaseModel):
        name: str
        comparison_method: ComparisonMethod
        comparison_target: float = None

    # fixme: the dataset is currently just being passed as-is into the completion
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": config.code_generation.prompt_planning_select_metric},
            {"role": "user", "content": f"Problem statement: {problem_statement}\nDataset: {dataset}"},
        ],
        response_format=ResponseFormat,
    )

    metric_response = completion.choices[0].message.parsed
    return Metric(
        metric_response.name, 0, MetricComparator(metric_response.comparison_method, metric_response.comparison_target)
    )


def select_stopping_condition(problem_statement: str, dataset: str, metric: Metric) -> StoppingCondition:
    """
    Selects the stopping condition for the given problem statement and dataset.

    :param problem_statement: definition of the problem
    :param dataset: data used for training and evaluation
    :param metric: the metric to optimise
    :return: the stopping condition
    """

    class ResponseFormat(BaseModel):
        max_generations: int
        max_time: int
        metric_threshold: float

    # fixme: the dataset is currently just being passed as-is into the completion
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": config.code_generation.prompt_planning_select_stopping_condition},
            {
                "role": "user",
                "content": f"Problem statement: {problem_statement}\nDataset: {dataset}\nMetric: {metric}",
            },
        ],
        response_format=ResponseFormat,
    )

    stopping_condition_response = completion.choices[0].message.parsed
    return StoppingCondition(
        max_generations=stopping_condition_response.max_generations,
        max_time=stopping_condition_response.max_time,
        metric=Metric(metric.name, stopping_condition_response.metric_threshold, metric.comparator),
    )


def generate_solution_plan(problem_statement: str, context: str = None) -> str:
    """
    Generates a solution plan for the given problem statement.

    :param problem_statement: definition of the problem
    :param context: additional context or memory for the solution
    :return: the generated solution plan
    """
    response = client.chat.completions.create(
        model="openai:gpt-4o",
        messages=[
            {"role": "system", "content": config.code_generation.prompt_planning_base},
            {
                "role": "user",
                "content": config.code_generation.prompt_planning_generate_plan.substitute(
                    problem_statement=problem_statement,
                    context=context,
                ),
            },
        ],
    )

    return response.choices[0].message.content


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
