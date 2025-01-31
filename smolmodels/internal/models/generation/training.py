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
from smolmodels.internal.common.providers.provider import Provider
from smolmodels.internal.common.utils.response import extract_code

logger = logging.getLogger(__name__)


class TrainingCodeGenerator:
    """
    A class to generate, fix, and review machine learning model training code.
    """

    def __init__(self, provider: Provider):
        """
        Initializes the TrainingCodeGenerator with an empty history.

        :param Provider provider: The provider to use for querying.
        """
        self.provider = provider
        self.history: List[Dict[str, str]] = []

    def generate_training_code(self, problem_statement: str, plan: str) -> str:
        """
        Generates machine learning model training code based on the given problem statement and solution plan.

        :param [str] problem_statement: The description of the problem to be solved.
        :param [str] plan: The proposed solution plan.
        :return str: The generated training code.
        """
        return extract_code(
            self.provider.query(
                system_message=config.code_generation.prompt_training_base.safe_substitute(),
                user_message=config.code_generation.prompt_training_generate.safe_substitute(
                    problem_statement=problem_statement,
                    plan=plan,
                    history=self.history,
                    allowed_packages=config.code_generation.allowed_packages,
                    training_data_path=config.execution.training_data_path,
                ),
            )
        )

    def fix_training_code(self, training_code: str, plan: str, review: str, problems: str = None) -> str:
        """
        Fixes the machine learning model training code based on the review and identified problems.

        :param [str] training_code: The previously generated training code.
        :param [str] plan: The proposed solution plan.
        :param [str] review: The review of the previous solution.
        :param [str] problems: Specific errors or bugs identified.
        :return str: The fixed training code.
        """

        class FixResponse(BaseModel):
            plan: str
            code: str

        response: FixResponse = FixResponse(
            **json.loads(
                self.provider.query(
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

    def review_training_code(self, training_code: str, problem_statement: str, plan: str, problems: str = None) -> str:
        """
        Reviews the machine learning model training code to identify improvements and fix issues.

        :param [str] training_code: The previously generated training code.
        :param [str] problem_statement: The description of the problem to be solved.
        :param [str] plan: The proposed solution plan.
        :param [str] problems: Specific errors or bugs identified.
        :return str: The review of the training code with suggestions for improvements.
        """
        return self.provider.query(
            system_message=config.code_generation.prompt_training_base.safe_substitute(),
            user_message=config.code_generation.prompt_training_review.safe_substitute(
                problem_statement=problem_statement,
                plan=plan,
                training_code=training_code,
                problems=problems,
                history=self.history,
                allowed_packages=config.code_generation.allowed_packages,
            ),
        )

    def generate_training_tests(self, problem_statement: str, plan: str, training_code: str) -> str:
        raise NotImplementedError("Generation of the training tests is not yet implemented.")

    def fix_training_tests(self, training_tests: str, training_code: str, review: str, problems: str = None) -> str:
        raise NotImplementedError("Fixing of the training tests is not yet implemented.")

    def review_training_tests(self, training_tests: str, training_code: str, problem_statement: str, plan: str) -> str:
        raise NotImplementedError("Review of the training tests is not yet implemented.")
