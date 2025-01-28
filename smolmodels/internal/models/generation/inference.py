import json
from typing import List, Dict
from pydantic import BaseModel

from smolmodels.config import config
from smolmodels.internal.common.providers.openai import OpenAIProvider
from smolmodels.internal.common.utils.response import extract_code

client = OpenAIProvider()


def generate_inference_code(input_schema: dict, output_schema: dict, training_code: str, context: str = None) -> str:
    """
    Generates inference code based on the problem statement, solution plan, and training code.

    :param [dict] input_schema: The schema of the input data.
    :param [dict] output_schema: The schema of the output data.
    :param [str] training_code: The training code that has already been generated.
    :param [str] context: Additional context or history.
    :return: The generated inference code.
    """
    return extract_code(
        client.query(
            system_message=config.code_generation.prompt_inference_base.safe_substitute(),
            user_message=config.code_generation.prompt_inference_generate.safe_substitute(
                input_schema=input_schema,
                output_schema=output_schema,
                training_code=training_code,
                context=context,
                allowed_packages=config.code_generation.allowed_packages,
            ),
        )
    )


def generate_inference_tests(problem_statement: str, plan: str, training_code: str, inference_code: str) -> str:
    """
    Generates tests for the inference code.

    Args:
        problem_statement (str): The description of the problem to be solved.
        plan (str): The proposed solution plan.
        training_code (str): The training code that has already been generated.
        inference_code (str): The generated inference code.

    Returns:
        str: The generated tests for the inference code.
    """
    raise NotImplementedError("Generation of the inference tests is not yet implemented.")


def fix_inference_code(inference_code: str, review: str, problems: str) -> str:
    """
    Fixes the inference code based on the review and identified problems.

    Args:
        inference_code (str): The previously generated inference code.
        review (str): The review of the previous solution.
        problems (str): Specific errors or bugs identified.

    Returns:
        str: The fixed inference code.
    """

    class FixResponse(BaseModel):
        plan: str
        code: str

    response: FixResponse = FixResponse(
        **json.loads(
            client.query(
                system_message=config.code_generation.prompt_inference_base.safe_substitute(),
                user_message=config.code_generation.prompt_inference_fix.safe_substitute(
                    inference_code=inference_code,
                    review=review,
                    problems=problems,
                ),
            )
        )
    )

    return extract_code(response.code)


def fix_inference_tests(inference_tests: str, inference_code: str, review: str, problems: str) -> str:
    """
    Fixes the tests for the inference code based on the review and identified problems.

    Args:
        inference_tests (str): The previously generated inference tests.
        inference_code (str): The previously generated inference code.
        review (str): The review of the previous solution.
        problems (str): Specific errors or bugs identified.

    Returns:
        str: The fixed inference tests.
    """
    raise NotImplementedError("Fixing of the inference tests is not yet implemented.")


def review_inference_code(
    inference_code: str,
    input_schema: dict,
    output_schema: dict,
    training_code: str,
    problems: str = None,
    context: str = None,
) -> str:
    """
    Reviews the inference code to identify improvements and fix issues.

    :param [str] inference_code: The previously generated inference code.
    :param [dict] input_schema: The schema of the input data.
    :param [dict] output_schema: The schema of the output data.
    :param [str] training_code: The training code that has already been generated.
    :param [str] problems: Specific errors or bugs identified.
    :param [str] context: Additional context or history.
    :return: The review of the inference code with suggestions for improvements.
    """
    return client.query(
        system_message=config.code_generation.prompt_inference_base.safe_substitute(),
        user_message=config.code_generation.prompt_inference_review.safe_substitute(
            inference_code=inference_code,
            training_code=training_code,
            problems=problems,
            context=context,
        ),
    )


def review_inference_tests(
    inference_tests: str, inference_code: str, problem_statement: str, plan: str, context: str = None
) -> str:
    """
    Reviews the tests for the inference code to identify improvements and fix issues.

    Args:
        inference_tests (str): The previously generated inference tests.
        inference_code (str): The previously generated inference code.
        problem_statement (str): The description of the problem to be solved.
        plan (str): The proposed solution plan.
        context (str, optional): Additional context or history. Defaults to None.

    Returns:
        str: The review of the inference tests with suggestions for improvements.
    """
    raise NotImplementedError("Review of the inference tests is not yet implemented.")


class InferenceCodeGenerator:
    def __init__(self):
        self.context: List[Dict[str, str]] = []

    def generate_inference_code(self, problem_statement: str, plan: str, training_code: str) -> str:
        """
        Generates inference code and updates the context.

        Args:
            problem_statement (str): The description of the problem to be solved.
            plan (str): The proposed solution plan.
            training_code (str): The training code that has already been generated.

        Returns:
            str: The generated inference code.
        """
        solution = generate_inference_code(problem_statement, plan, training_code, str(self.context))
        self.context.append(
            {
                "problem_statement": problem_statement,
                "plan": plan,
                "training_code": training_code,
                "solution": solution,
            }
        )
        return solution

    def fix_inference_code(self, inference_code: str, review: str, problems: str) -> str:
        """
        Fixes inference code and updates the context.

        Args:
            inference_code (str): The previously generated inference code.
            review (str): The review of the previous solution.
            problems (str): Specific errors or bugs identified.

        Returns:
            str: The fixed inference code.
        """
        solution = fix_inference_code(inference_code, review, problems)
        self.context.append(
            {
                "inference_code": inference_code,
                "review": review,
                "problems": problems,
                "solution": solution,
            }
        )
        return solution

    def review_inference_code(self, inference_code: str, problem_statement: str, plan: str) -> str:
        """
        Reviews inference code and updates the context.

        Args:
            inference_code (str): The previously generated inference code.
            problem_statement (str): The description of the problem to be solved.
            plan (str): The proposed solution plan.

        Returns:
            str: The review of the inference code with suggestions for improvements.
        """
        review = review_inference_code(inference_code, problem_statement, plan, str(self.context))
        self.context.append(
            {
                "inference_code": inference_code,
                "problem_statement": problem_statement,
                "plan": plan,
                "review": review,
            }
        )
        return review
