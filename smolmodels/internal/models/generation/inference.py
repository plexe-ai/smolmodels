# todo: algorithms for generating the inference code
from typing import List, Dict
from smolmodels.config import config

import openai

client = openai.OpenAI()


def generate_inference_code(problem_statement: str, plan: str, training_code: str, context: str = None) -> str:
    """
    Generates inference code based on the problem statement, solution plan, and training code.

    Args:
        problem_statement (str): The description of the problem to be solved.
        plan (str): The proposed solution plan.
        training_code (str): The training code that has already been generated.
        context (str, optional): Additional context or history. Defaults to None.

    Returns:
        str: The generated inference code.
    """
    response = client.chat.completions.create(
        model="openai:gpt-4o",
        messages=[
            {"role": "system", "content": config.code_generation.prompt_inference_base},
            {
                "role": "user",
                "content": config.code_generation.prompt_inference_generate.substitute(
                    problem_statement=problem_statement,
                    plan=plan,
                    training_code=training_code,
                    context=context,
                ),
            },
        ],
    )

    return response.choices[0].message.content


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
    response = client.chat.completions.create(
        model="openai:gpt-4o",
        messages=[
            {"role": "system", "content": config.code_generation.prompt_inference_base},
            {
                "role": "user",
                "content": config.code_generation.prompt_inference_fix.substitute(
                    inference_code=inference_code,
                    review=review,
                    problems=problems,
                ),
            },
        ],
    )

    return response.choices[0].message.content


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


def review_inference_code(inference_code: str, problem_statement: str, plan: str, context: str = None) -> str:
    """
    Reviews the inference code to identify improvements and fix issues.

    Args:
        inference_code (str): The previously generated inference code.
        problem_statement (str): The description of the problem to be solved.
        plan (str): The proposed solution plan.
        context (str, optional): Additional context or history. Defaults to None.

    Returns:
        str: The review of the inference code with suggestions for improvements.
    """
    response = client.chat.completions.create(
        model="openai:gpt-4o",
        messages=[
            {"role": "system", "content": config.code_generation.prompt_inference_base},
            {
                "role": "user",
                "content": config.code_generation.prompt_inference_review.substitute(
                    problem_statement=problem_statement,
                    plan=plan,
                    inference_code=inference_code,
                    context=context,
                ),
            },
        ],
    )

    return response.choices[0].message.content


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
