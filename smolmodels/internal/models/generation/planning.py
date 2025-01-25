from typing import List, Dict
import aisuite as ai
import openai
from pydantic import BaseModel


from smolmodels.internal.models.entities.metric import Metric, MetricComparator, ComparisonMethod

import logging

from smolmodels.internal.models.entities.stopping_condition import StoppingCondition

logger = logging.getLogger(__name__)


ALLOWED_PACKAGES = [
    "numpy",
    "pandas",
    "scikit-learn",
]


def select_metric_to_optimise(problem_statement: str, dataset: str) -> Metric:
    class ResponseFormat(BaseModel):
        name: str
        comparison_method: ComparisonMethod
        comparison_target: float = None

    client = openai.OpenAI()
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06", messages=[], response_format=ResponseFormat
    )

    metric_response = completion.choices[0].message.parsed
    return Metric(
        metric_response.name, 0, MetricComparator(metric_response.comparison_method, metric_response.comparison_target)
    )


def select_stopping_condition(problem_statement: str, dataset: str) -> StoppingCondition:
    # todo: implement this function
    return StoppingCondition(
        max_generations=10,
        max_time=3600,
        metric=Metric("accuracy", 0.9, MetricComparator(ComparisonMethod.HIGHER_IS_BETTER)),
    )


def generate_solution_plan(problem_statement: str, context: str = None) -> str:
    client = ai.Client()

    system_prompt = (
        "You are an experienced Machine Learning Engineer attending a Kaggle competition. In order to win this competition, "
        "you need to come up with an excellent and creative plan for a solution"
    )
    prompt: str = (
        "We will now provide a description of the task. Write a plan for a solution to the following task."
        "\n\n"
        f"**Task description:** {problem_statement}"
        "\n\n"
        f"**Memory:** {context}"
        "\n\n"
        "Your response should be a brief outline/sketch of your proposed solution in natural language (3-5 sentences), "
        "followed by a single markdown code block (wrapped in ```) which implements this solution and prints out the evaluation metric. "
        "There should be no additional headings or text in your response. Just natural language text followed by a newline and then the markdown code block. "
        "\n\n"
        "This first solution design should be relatively simple, without ensembling or hyper-parameter optimization."
        "Take the Memory section into consideration when proposing the design"
        " don't propose the same modelling solution but keep the evaluation the same."
        "The solution sketch should be 3-5 sentences."
        "Propose an evaluation metric that is reasonable for this task."
        "Don't suggest to do EDA."
        "\n\n"
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
    def __init__(self):
        self.context: List[Dict[str, str]] = []

    def generate_solution_plan(self, problem_statement: str) -> str:
        solution = generate_solution_plan(problem_statement, str(self.context))
        self.context.append(
            {
                "problem_statement": problem_statement,
                "plan": solution,
            }
        )
        return solution
