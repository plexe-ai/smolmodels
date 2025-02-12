# smolmodels/internal/models/generation/inference.py

"""
This module provides functionality for generating inference code for machine learning models.
"""

import json
from typing import List, Dict

from pydantic import BaseModel

from smolmodels.config import config
from smolmodels.internal.common.provider import Provider
from smolmodels.internal.common.utils.response import extract_code


class InferenceCodeGenerator:
    def __init__(self, provider: Provider):
        """
        Initializes the InferenceCodeGenerator with an empty context.
        :param provider: the LLM provider to use for querying
        """
        self.provider: Provider = provider
        self.context: List[Dict[str, str]] = []

    def generate_inference_code(
        self, input_schema: dict, output_schema: dict, training_code: str, model_id: str
    ) -> str:
        """
        Generates inference code based on the problem statement, solution plan, and training code.

        :param [dict] input_schema: The schema of the input data.
        :param [dict] output_schema: The schema of the output data.
        :param [str] training_code: The training code that has already been generated.
        :param [str] model_id: The ID of the model to load.
        :return: The generated inference code.
        """
        return extract_code(
            self.provider.query(
                system_message=config.code_generation.prompt_inference_base.safe_substitute(),
                user_message=config.code_generation.prompt_inference_generate.safe_substitute(
                    input_schema=input_schema,
                    output_schema=output_schema,
                    training_code=training_code,
                    model_id=model_id,
                    context="",  # todo: implement memory to provide as 'context'
                    allowed_packages=config.code_generation.allowed_packages,
                ),
            )
        )

    def fix_inference_code(self, inference_code: str, review: str, problems: str) -> str:
        """
        Fixes the inference code based on the review and identified problems.

        :param [str] inference_code: The previously generated inference code.
        :param [str] review: The review of the previous solution.
        :param [str] problems: Specific errors or bugs identified.
        :return str: The fixed inference code.
        """

        class FixResponse(BaseModel):
            plan: str
            code: str

        response: FixResponse = FixResponse(
            **json.loads(
                self.provider.query(
                    system_message=config.code_generation.prompt_inference_base.safe_substitute(),
                    user_message=config.code_generation.prompt_inference_fix.safe_substitute(
                        inference_code=inference_code,
                        review=review,
                        problems=problems,
                    ),
                    response_format=FixResponse,
                )
            )
        )
        return extract_code(response.code)

    def review_inference_code(
        self, inference_code: str, input_schema: dict, output_schema: dict, training_code: str, problems: str = None
    ) -> str:
        """
        Reviews the inference code to identify improvements and fix issues.

        :param [str] inference_code: The previously generated inference code.
        :param [dict] input_schema: The schema of the input data.
        :param [dict] output_schema: The schema of the output data.
        :param [str] training_code: The training code that has already been generated.
        :param [str] problems: Specific errors or bugs identified.
        :return: The review of the inference code with suggestions for improvements.
        """
        return self.provider.query(
            system_message=config.code_generation.prompt_inference_base.safe_substitute(),
            user_message=config.code_generation.prompt_inference_review.safe_substitute(
                inference_code=inference_code,
                input_schema=input_schema,
                output_schema=output_schema,
                training_code=training_code,
                problems=problems,
                context="",  # todo: implement memory to provide as 'context'
            ),
        )

    def generate_inference_tests(
        self, problem_statement: str, plan: str, training_code: str, inference_code: str
    ) -> str:
        raise NotImplementedError("Generation of the inference tests is not yet implemented.")

    def fix_inference_tests(self, inference_tests: str, inference_code: str, review: str, problems: str) -> str:
        raise NotImplementedError("Fixing of the inference tests is not yet implemented.")

    def review_inference_tests(
        self, inference_tests: str, inference_code: str, problem_statement: str, plan: str
    ) -> str:
        raise NotImplementedError("Review of the inference tests is not yet implemented.")
