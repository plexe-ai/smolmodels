# smolmodels/internal/models/generation/planning.py

"""
This module provides functions and classes for generating and planning solutions for machine learning problems.
"""

import logging
from typing import List, Dict

import aisuite as ai
import openai
from pydantic import BaseModel

from smolmodels.internal.models.entities.metric import Metric, MetricComparator, ComparisonMethod
from smolmodels.internal.models.entities.stopping_condition import StoppingCondition

logger = logging.getLogger(__name__)

ALLOWED_PACKAGES = [
    "numpy",
    "pandas",
    "scikit-learn",
]


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

    client = openai.OpenAI()

    # fixme: the dataset is currently just being passed as-is into the completion
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": "Select the metric to optimise for the given problem statement and dataset."},
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

    client = openai.OpenAI()

    # fixme: the dataset is currently just being passed as-is into the completion
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": "Select the stopping condition for the given problem statement and dataset."},
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
    client = ai.Client()

    system_prompt = (
        "You are an experienced Machine Learning Engineer attending a Kaggle competition. In order to win this competition, "
        "you need to come up with an excellent and creative plan for a solution"
    )
    prompt: str = (
        "We will now provide a description of the task. Write a plan for a solution to the following task.\n\n"
        f"**Task description:** {problem_statement}\n\n"
        f"**Memory:** {context}\n\n"
        "Your response should be a brief outline/sketch of your proposed solution in natural language (3-5 sentences), "
        "followed by a single markdown code block (wrapped in ```) which implements this solution and prints out the evaluation metric. "
        "There should be no additional headings or text in your response. Just natural language text followed by a newline and then the markdown code block.\n\n"
        "This first solution design should be relatively simple, without ensembling or hyper-parameter optimization."
        "Take the Memory section into consideration when proposing the design"
        " don't propose the same modelling solution but keep the evaluation the same."
        "The solution sketch should be 3-5 sentences."
        "Propose an evaluation metric that is reasonable for this task."
        "Don't suggest to do EDA.\n\n"
        f"Your solution can only use the following ML frameworks: {ALLOWED_PACKAGES}."
    )

    response = client.chat.completions.create(
        model="openai:gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
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
