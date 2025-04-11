"""
This module defines agent tools for evaluating the properties and performance of models.
"""

import logging

from smolagents import tool

from smolmodels.internal.common.provider import Provider
from smolmodels.internal.models.generation.review import ModelReviewer

logger = logging.getLogger(__name__)


@tool
def review_model(intent: str, solution_plan: str, training_code: str, inference_code: str, provider: str) -> dict:
    """
    Reviews the entire model and extracts metadata.

    Args:
        intent: The model intent
        solution_plan: The solution plan that was implemented
        training_code: The training code that was used
        inference_code: The inference code that was generated
        provider: The string identifier representing the model that should be used, e.g. 'openai/gpt-4o'

    Returns:
        A dictionary containing model metadata
    """
    reviewer = ModelReviewer(Provider(provider))
    return reviewer.review_model(intent, solution_plan, training_code, inference_code)
