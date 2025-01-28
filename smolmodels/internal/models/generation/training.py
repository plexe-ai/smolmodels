"""
This module provides functions and classes for generating, fixing, and reviewing machine learning model training code.

Functions:
    generate_training_code: Generates machine learning model training code based on a problem statement and solution plan.
    generate_training_tests: Generates tests for the machine learning model training code.
    fix_training_code: Fixes the machine learning model training code based on review and identified problems.
    fix_training_tests: Fixes the tests for the machine learning model training code based on review and identified problems.
    review_training_code: Reviews the machine learning model training code to identify improvements and fix issues.
    review_training_tests: Reviews the tests for the machine learning model training code to identify improvements and fix issues.

Classes:
    TrainingCodeGenerator: A class to generate, fix, and review machine learning model training code.
"""

import json
import logging
from typing import List, Dict

from pydantic import BaseModel

from smolmodels.config import config
from smolmodels.internal.common.providers.openai import OpenAIProvider
from smolmodels.internal.common.utils.response import extract_code

logger = logging.getLogger(__name__)

client = OpenAIProvider()


def generate_training_code(problem_statement: str, plan: str, history: str = None) -> str:
    """
    Generates machine learning model training code based on the given problem statement and solution plan.

    Args:
        problem_statement (str): The description of the problem to be solved.
        plan (str): The proposed solution plan.
        history (str, optional): The history of previous attempts or context. Defaults to None.

    Returns:
        str: The generated training code.
    """
    return extract_code(
        client.query(
            system_message=config.code_generation.prompt_training_base.safe_substitute(),
            user_message=config.code_generation.prompt_training_generate.safe_substitute(
                problem_statement=problem_statement,
                plan=plan,
                history=history,
                allowed_packages=config.code_generation.allowed_packages,
                training_data_path=config.execution.training_data_path,
            ),
        )
    )


def generate_training_tests(problem_statement: str, plan: str, training_code: str) -> str:
    """
    Generates tests for the machine learning model training code.

    Args:
        problem_statement (str): The description of the problem to be solved.
        plan (str): The proposed solution plan.
        training_code (str): The generated training code.

    Returns:
        str: The generated tests for the training code.
    """
    raise NotImplementedError("Generation of the training tests is not yet implemented.")


def fix_training_code(training_code: str, plan: str, review: str, problems: str = None, history: str = None) -> str:
    """
    Fixes the machine learning model training code based on the review and identified problems.

    Args:
        training_code (str): The previously generated training code.
        plan (str): The proposed solution plan.
        review (str): The review of the previous solution.
        problems (str, optional): Specific errors or bugs identified. Defaults to None.
        history (str, optional): The history of previous attempts or context. Defaults to None.

    Returns:
        str: The fixed training code.
    """

    class FixResponse(BaseModel):
        plan: str
        code: str

    response: FixResponse = FixResponse(
        **json.loads(
            client.query(
                system_message=config.code_generation.prompt_training_base.safe_substitute(),
                user_message=config.code_generation.prompt_training_fix.safe_substitute(
                    plan=plan,
                    training_code=training_code,
                    review=review,
                    problems=problems,
                    training_data_path=config.execution.training_data_path,
                    allowed_packages=config.code_generation.allowed_packages,
                ),
                response_format=FixResponse,
            )
        )
    )
    return extract_code(response.code)


def fix_training_tests(training_tests: str, training_code: str, review: str, problems: str = None) -> str:
    """
    Fixes the tests for the machine learning model training code based on the review and identified problems.

    Args:
        training_tests (str): The previously generated training tests.
        training_code (str): The previously generated training code.
        review (str): The review of the previous solution.
        problems (str, optional): Specific errors or bugs identified. Defaults to None.

    Returns:
        str: The fixed training tests.
    """
    raise NotImplementedError("Fixing of the training tests is not yet implemented.")


def review_training_code(
    training_code: str, problem_statement: str, plan: str, problems: str = None, history: str = None
) -> str:
    """
    Reviews the machine learning model training code to identify improvements and fix issues.

    Args:
        training_code (str): The previously generated training code.
        problem_statement (str): The description of the problem to be solved.
        plan (str): The proposed solution plan.
        problems (str, optional): Specific errors or bugs identified. Defaults to None.
        history (str, optional): The history of previous attempts or context. Defaults to None.

    Returns:
        str: The review of the training code with suggestions for improvements.
    """
    return client.query(
        system_message=config.code_generation.prompt_training_base.safe_substitute(),
        user_message=config.code_generation.prompt_training_review.safe_substitute(
            problem_statement=problem_statement,
            plan=plan,
            training_code=training_code,
            problems=problems,
            history=history,
            allowed_packages=config.code_generation.allowed_packages,
        ),
    )


def review_training_tests(
    training_tests: str, training_code: str, problem_statement: str, plan: str, context: str = None
) -> str:
    """
    Reviews the tests for the machine learning model training code to identify improvements and fix issues.

    Args:
        training_tests (str): The previously generated training tests.
        training_code (str): The previously generated training code.
        problem_statement (str): The description of the problem to be solved.
        plan (str): The proposed solution plan.
        context (str, optional): Additional context or history. Defaults to None.

    Returns:
        str: The review of the training tests with suggestions for improvements.
    """
    raise NotImplementedError("Review of the training tests is not yet implemented.")


class TrainingCodeGenerator:
    """
    A class to generate, fix, and review machine learning model training code.
    """

    def __init__(self):
        """
        Initializes the TrainingCodeGenerator with an empty history.
        """
        self.history: List[Dict[str, str]] = []

    def generate_training_code(self, problem_statement: str, plan: str) -> str:
        """
        Generates machine learning model training code and updates the history.

        Args:
            problem_statement (str): The description of the problem to be solved.
            plan (str): The proposed solution plan.

        Returns:
            str: The generated training code.
        """
        solution = generate_training_code(problem_statement, plan, str(self.history))
        self.history.append({"problem_statement": problem_statement, "plan": plan, "solution": solution})
        return solution

    def fix_training_code(self, training_code: str, plan: str, review: str, problems: str = None) -> str:
        """
        Fixes the machine learning model training code and updates the history.

        Args:
            training_code (str): The previously generated training code.
            plan (str): The proposed solution plan.
            review (str): The review of the previous solution.
            problems (str, optional): Specific errors or bugs identified. Defaults to None.

        Returns:
            str: The fixed training code.
        """
        solution = fix_training_code(training_code, plan, review, problems, str(self.history))
        self.history.append(
            {"training_code": training_code, "review": review, "problems": problems, "solution": solution}
        )
        return solution

    def review_training_code(self, training_code: str, problem_statement: str, plan: str, problems: str = None) -> str:
        """
        Reviews the machine learning model training code and updates the history.

        Args:
            training_code (str): The previously generated training code.
            problem_statement (str): The description of the problem to be solved.
            plan (str): The proposed solution plan.
            problems (str, optional): Specific errors or bugs identified. Defaults to None.

        Returns:
            str: The review of the training code with suggestions for improvements.
        """
        review = review_training_code(training_code, problem_statement, plan, problems, str(self.history))
        self.history.append(
            {
                "training_code": training_code,
                "problem_statement": problem_statement,
                "plan": plan,
                "problems": problems,
                "review": review,
            }
        )
        return review
