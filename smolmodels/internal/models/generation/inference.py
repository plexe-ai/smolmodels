# todo: algorithms for generating the inference code
from typing import List, Dict


def generate_inference_code(problem_statement: str, plan: str, training_code: str, context: str = None) -> str:
    raise NotImplementedError("Generation of the inference code is not yet implemented.")


def generate_inference_tests(problem_statement: str, plan: str, training_code: str, inference_code: str) -> str:
    raise NotImplementedError("Generation of the inference tests is not yet implemented.")


def fix_inference_code(inference_code: str, review: str, problems: str) -> str:
    raise NotImplementedError("Fixing of the inference code is not yet implemented.")


def fix_inference_tests(inference_tests: str, inference_code: str, review: str, problems: str) -> str:
    raise NotImplementedError("Fixing of the inference tests is not yet implemented.")


def review_inference_code(inference_code: str, problem_statement: str, plan: str, context: str = None) -> str:
    raise NotImplementedError("Review of the inference code is not yet implemented.")


def review_inference_tests(
    inference_tests: str, inference_code: str, problem_statement: str, plan: str, context: str = None
) -> str:
    raise NotImplementedError("Review of the inference tests is not yet implemented.")


class InferenceCodeGenerator:
    def __init__(self):
        self.context: List[Dict[str, str]] = []

    def generate_inference_code(self, problem_statement: str, plan: str, training_code: str) -> str:
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
