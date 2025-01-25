# todo: algorithms for generating the inference code
from typing import List, Dict

import openai

ALLOWED_PACKAGES = [
    "numpy",
    "pandas",
    "scikit-learn",
]


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
    client = openai.OpenAI()

    system_prompt = (
        "You are an experienced Machine Learning Engineer tasked with deploying a trained machine learning model. "
        "You must write an inference script based on a given problem statement, solution plan, and training code. "
        "The script should take in new data, preprocess it appropriately, load the trained model, and return predictions. "
        "Make sure to handle common errors gracefully."
    )
    prompt = (
        f"# Task description: {problem_statement}\n"
        f"# Solution plan: {plan}\n"
        f"# Training code:\n{training_code}\n"
        f"# Context: {context if context else 'N/A'}\n"
        "\n"
        "Write a Python script for inference that:\n"
        "1. Loads the trained model saved in 'model.joblib' from the './working' directory.\n"
        "2. Takes a new input dataset, preprocesses it as per the training pipeline, and generates predictions.\n"
        "3. Outputs the predictions as a CSV file named 'predictions.csv' in the './working' directory.\n"
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
    client = openai.OpenAI()

    system_prompt = "You are an experienced Machine Learning Engineer tasked with fixing an inference script."
    prompt = (
        f"# Original Inference Code:\n{inference_code}\n"
        f"# Review of Issues:\n{review}\n"
        f"# Specific Errors/Bugs:\n{problems}\n"
        "\n"
        "Revise the inference code to address the identified issues and ensure it performs as expected. "
        "The code should handle common errors gracefully and ensure compatibility with the training pipeline."
    )

    response = client.chat.completions.create(
        model="openai:gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
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
    client = openai.OpenAI()

    system_prompt = "You are an experienced Machine Learning Engineer tasked with reviewing an inference script."
    prompt = (
        f"# Problem Statement:\n{problem_statement}\n"
        f"# Solution Plan:\n{plan}\n"
        f"# Inference Code:\n{inference_code}\n"
        f"# Context:\n{context if context else 'N/A'}\n"
        "\n"
        "Review the inference code to identify potential improvements, ensure it aligns with the solution plan, and "
        "handles inputs and outputs as expected. Provide suggestions for optimisation and error handling."
    )

    response = client.chat.completions.create(
        model="openai:gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
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
