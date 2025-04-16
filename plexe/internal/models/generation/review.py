"""
This module provides functionality for reviewing and analyzing generated models.

It examines the solution plan, training code, and inference code to extract
metadata about the model, such as the framework used, model type, and provides
explanations about how the model works and why it's appropriate for the task.
"""

import json
import logging
from datetime import datetime
from typing import Dict

from pydantic import BaseModel

from plexe.config import prompt_templates
from plexe.internal.common.provider import Provider

logger = logging.getLogger(__name__)


class ModelReviewResponse(BaseModel):
    """
    Response model for the model review operation.
    """

    framework: str
    model_type: str
    explanation: str


class ModelReviewer:
    """
    A class for analyzing and reviewing generated models.
    """

    def __init__(self, provider: Provider):
        """
        Initialize the model reviewer with a provider.

        :param provider: The provider to use for generating model reviews
        """
        self.provider = provider

    def review_model(self, intent: str, solution_plan: str, training_code: str, inference_code: str) -> Dict[str, str]:
        """
        Review a generated model to extract metadata and explanations.

        :param intent: The original model intent
        :param solution_plan: The solution plan used to generate the model
        :param training_code: The generated training code
        :param inference_code: The generated inference code
        :return: A dictionary containing framework, model_type, explanation, and creation_date
        """
        try:
            response = self.provider.query(
                system_message=prompt_templates.review_system(),
                user_message=prompt_templates.review_model(
                    intent=intent,
                    solution_plan=solution_plan,
                    training_code=training_code,
                    inference_code=inference_code,
                ),
                response_format=ModelReviewResponse,
            )

            # Parse the response and create metadata dictionary
            review_data = json.loads(response)

            # Create metadata dictionary with review results and creation date
            metadata = {
                "framework": review_data["framework"],
                "model_type": review_data["model_type"],
                "selection_rationale": review_data["explanation"],
                "creation_date": datetime.now().isoformat(),
            }
            return metadata

        except Exception as e:
            logger.warning(f"Error during model review: {str(e)}")
            # Return default values if there was an error
            return {
                "framework": "Unknown",
                "model_type": "Unknown",
                "selection_rationale": "Could not determine model details due to an error.",
                "creation_date": datetime.now().isoformat(),
            }
