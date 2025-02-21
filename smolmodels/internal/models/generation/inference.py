# smolmodels/internal/models/generation/inference.py

"""
This module provides functionality for generating inference code for machine learning models.
"""

import json
from typing import List, Dict
from pathlib import Path

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

    def _generate_model_loading(self, training_code: str, model_dir: Path) -> str:
        """
        Generate code for loading the trained model files.

        :param training_code: The training code to analyze for model saving patterns
        :param model_dir: Directory containing the model files
        :return: Code snippet for model loading
        """
        return extract_code(
            self.provider.query(
                system_message=config.code_generation.prompt_inference_base.safe_substitute(),
                user_message=config.code_generation.prompt_inference_model_loading.safe_substitute(
                    training_code=training_code
                ),
            )
        )

    def _generate_preprocessing(self, input_schema: dict, training_code: str) -> str:
        """
        Generate code for preprocessing input data before prediction.

        :param input_schema: Schema defining the input data format
        :param training_code: Training code to analyze for preprocessing steps
        :return: Code snippet for preprocessing
        """
        return extract_code(
            self.provider.query(
                system_message=config.code_generation.prompt_inference_base.safe_substitute(),
                user_message=config.code_generation.prompt_inference_preprocessing.safe_substitute(
                    input_schema=input_schema, training_code=training_code
                ),
            )
        )

    def _generate_prediction(
        self, output_schema: dict, training_code: str, model_loading_code: str, preprocessing_code: str
    ) -> str:
        """
        Generate code for making predictions with the loaded model.

        :param output_schema: Schema defining the expected output format
        :param training_code: Training code to analyze for prediction patterns
        :param model_loading_code: Generated code for loading model files
        :param preprocessing_code: Generated code for preprocessing input data
        :return: Code snippet for prediction
        """
        return extract_code(
            self.provider.query(
                system_message=config.code_generation.prompt_inference_base.safe_substitute(),
                user_message=config.code_generation.prompt_inference_prediction.safe_substitute(
                    output_schema=output_schema,
                    training_code=training_code,
                    model_loading_code=model_loading_code,
                    preprocessing_code=preprocessing_code,
                ),
            )
        )

    def _combine_code_stages(
        self, model_loading_code: str, preprocessing_code: str, prediction_code: str, model_dir: Path
    ) -> str:
        """
        Combine the separately generated code stages into a complete inference script.

        :param model_loading_code: Code for loading model files
        :param preprocessing_code: Code for preprocessing input data
        :param prediction_code: Code for making predictions
        :param model_dir: Directory containing model files
        :return: Complete inference script
        """
        return extract_code(
            self.provider.query(
                system_message=config.code_generation.prompt_inference_base.safe_substitute(),
                user_message=config.code_generation.prompt_inference_combine.safe_substitute(
                    model_loading_code=model_loading_code,
                    preprocessing_code=preprocessing_code,
                    prediction_code=prediction_code,
                    model_dir=model_dir.as_posix(),
                ),
            )
        )

    def generate_inference_code(
        self, input_schema: dict, output_schema: dict, training_code: str, filedir: Path
    ) -> str:
        """
        Generates inference code based on the problem statement, solution plan, and training code.

        :param input_schema: The schema of the input data.
        :param output_schema: The schema of the output data.
        :param training_code: The training code that has already been generated.
        :param filedir: The directory in which the predictor should expect model files.
        :return: The generated inference code.
        """
        model_dir = Path(filedir).resolve()

        # Stage 1: Generate model loading code
        model_loading_code = self._generate_model_loading(training_code, model_dir)

        # Stage 2: Generate preprocessing code
        preprocessing_code = self._generate_preprocessing(input_schema, training_code)

        # Stage 3: Generate prediction code with context from previous stages
        prediction_code = self._generate_prediction(
            output_schema, training_code, model_loading_code, preprocessing_code
        )

        # Combine the stages
        return self._combine_code_stages(model_loading_code, preprocessing_code, prediction_code, model_dir)

    def fix_inference_code(self, inference_code: str, review: str, problems: str, filedir: Path) -> str:
        """
        Fixes the inference code based on the review and identified problems.

        :param [str] inference_code: The previously generated inference code.
        :param [str] review: The review of the previous solution.
        :param [str] problems: Specific errors or bugs identified.
        :param [str] filedir: The directory in which the predictor should expect model files.
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
                        fildeir=filedir.as_posix(),
                    ),
                    response_format=FixResponse,
                )
            )
        )
        return extract_code(response.code)

    def review_inference_code(
        self,
        inference_code: str,
        input_schema: dict,
        output_schema: dict,
        training_code: str,
        problems: str = None,
        filedir: Path = None,
    ) -> str:
        """
        Reviews the inference code to identify improvements and fix issues.

        :param [str] inference_code: The previously generated inference code.
        :param [dict] input_schema: The schema of the input data.
        :param [dict] output_schema: The schema of the output data.
        :param [str] training_code: The training code that has already been generated.
        :param [str] problems: Specific errors or bugs identified.
        :param [str] filedir: The directory in which the predictor should expect model files.
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
                filedir=filedir.as_posix(),
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
