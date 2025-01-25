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

Constants:
    ALLOWED_PACKAGES: List of allowed packages for generating training code.
"""

from typing import List, Dict

import openai
from pydantic import BaseModel

ALLOWED_PACKAGES = [
    "numpy",
    "pandas",
    "scikit-learn",
]


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
    client = openai.OpenAI()

    system_prompt = (
        "You are an experienced Machine Learning Engineer working on a Kaggle competition. "
        "To win this competition, you must write an excellent machine learning model training code, "
        "based on a given problem statement and solution plan. Your training code should implement the "
        "proposed solution plan."
    )
    prompt = (
        "Write a Python script that trains a machine learning model to solve the following task, "
        "following the outlined solution plan.\n\n"
        f"# Task description: {problem_statement}\n\n"
        f"# Solution plan: {plan}\n\n"
        f"# History: {history if history is not None else "N/A"}\n\n"
        "The code should **implement the proposed solution** and **print the value of the evaluation metric "
        "computed on a hold-out validation set**. Train a new model and save it as 'model.joblib' in the './working' "
        "directory using `joblib.dump()`. Feel free to produce any other joblib files. **DO NOT SAVE any "
        "preprocessors in the same joblib file using keys**. ALWAYS use separate joblib files for different objects. "
        "No parts of the code should be skipped; don't terminate before finishing the script. Your response should "
        "only contain a single code block. All the provided input data is stored in the './input' directory. "
        "You can use the './working' directory to store any temporary files that your code needs to create.\n\n"
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
    client = openai.OpenAI()

    system_prompt = "You are an experienced Machine Learning Engineer working on a Kaggle competition. "
    prompt = (
        "Your previous solution had a bug, so based on the information below, you should revise it in order "
        "to fix this bug. Your response should consist of an implementation outline plan in natural language, "
        "and a rewritten version of the code in which all highlighted problems are fixed.\n\n"
        f"# Original Solution Plan:\n{plan}\n\n"
        f"# Previous Solution:\n{training_code}\n\n"
        f"# Review of Issues:\n{review}\n\n"
        f"# Specific Errors/Bugs:\n{problems if problems is not None else "N/A"}\n\n"
        f"# History:\n{history if history is not None else "N/A"}\n\n"
        "The code should **implement the proposed solution** and **print the value of the evaluation metric "
        "computed on a hold-out validation set**. Train a new model and save it as 'model.joblib' in the './working' "
        "directory using `joblib.dump()`. Feel free to produce any other joblib files. **DO NOT SAVE any "
        "preprocessors in the same joblib file using keys**. ALWAYS use separate joblib files for different objects. "
        "No parts of the code should be skipped; don't terminate before finishing the script. Your response should "
        "only contain a single code block. All the provided input data is stored in the './input' directory. "
        "You can use the './working' directory to store any temporary files that your code needs to create.\n\n"
        f"Your solution can only use the following ML frameworks: {ALLOWED_PACKAGES}."
    )

    class ResponseFormat(BaseModel):
        plan: str
        code: str

    response = client.beta.chat.completions.parse(
        model="openai:gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        response_format=ResponseFormat,
    )

    return response.choices[0].message.content


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
    client = openai.OpenAI()

    system_prompt = "You are an experienced Machine Learning Engineer working on a Kaggle competition. "
    prompt = (
        "You are provided with a previously developed solution below, and should review it in order to further "
        "increase the (test time) performance and fix any highlighted issues/bugs. Write a review of the code, "
        "any issues that should be fixed, and any other suggestions for how the solution could be improved while "
        "still implementing the original plan.\n\n"
        f"# Problem Statement:\n{problem_statement}\n\n"
        f"# Original Solution Plan:\n{plan}\n\n"
        f"# Previous Solution:\n{training_code}\n\n"
        f"# Specific Errors/Bugs:\n{problems if problems is not None else "N/A"}\n\n"
        f"# History:\n{history if history is not None else "N/A"}\n\n"
        "The solution sketch should be a brief natural language description of how the previous solution "
        "can be improved. You should be very specific and should only propose a single actionable improvement. "
        "This improvement should be atomic so that we can experimentally evaluate the effect of the proposed change. "
        "Take the 'history' section of previous reviews into consideration when proposing the improvement. The "
        "review should be 3-5 sentences. Don't suggest to do EDA.\n\n"
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
