# todo: algorithms for generating the model training code


from typing import List, Dict

ALLOWED_PACKAGES = [
    "numpy",
    "pandas",
    "scikit-learn",
]


def generate_training_code(problem_statement: str, plan: str, context: str = None) -> str:
    raise NotImplementedError("Generation of the training code is not yet implemented.")


def generate_training_tests(problem_statement: str, plan: str, training_code: str) -> str:
    raise NotImplementedError("Generation of the training tests is not yet implemented.")


def fix_training_code(training_code: str, review: str, problems: str = None) -> str:
    raise NotImplementedError("Fixing of the training code is not yet implemented.")


def fix_training_tests(training_tests: str, training_code: str, review: str, problems: str = None) -> str:
    raise NotImplementedError("Fixing of the training tests is not yet implemented.")


def review_training_code(training_code: str, problem_statement: str, plan: str, context: str = None) -> str:
    raise NotImplementedError("Review of the training code is not yet implemented.")


def review_training_tests(
    training_tests: str, training_code: str, problem_statement: str, plan: str, context: str = None
) -> str:
    raise NotImplementedError("Review of the training tests is not yet implemented.")


class TrainingCodeGenerator:
    def __init__(self):
        self.context: List[Dict[str, str]] = []

    def generate_training_code(self, problem_statement: str, plan: str) -> str:
        solution = generate_training_code(problem_statement, plan, str(self.context))
        self.context.append({"problem_statement": problem_statement, "plan": plan, "solution": solution})
        return solution
