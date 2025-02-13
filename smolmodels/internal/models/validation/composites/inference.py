# internal/models/validation/composites/inference.py

"""
This module defines a composite validator for validating the correctness of prediction code.

Classes:
    - InferenceCodeValidator: A validator class that validates the correctness of prediction code.
"""

from smolmodels.internal.common.provider import Provider
from smolmodels.internal.models.validation.primitives.predict import PredictorValidator
from smolmodels.internal.models.validation.primitives.syntax import SyntaxValidator
from smolmodels.internal.models.validation.composite import CompositeValidator


class InferenceCodeValidator(CompositeValidator):
    """
    A validator class that validates the correctness of prediction code.
    """

    def __init__(self, provider: Provider, intent: str, input_schema: dict, output_schema: dict, n_samples=10):
        """
        Initialize the PredictionValidator with the name 'prediction'.
        """
        super().__init__(
            "prediction",
            [SyntaxValidator(), PredictorValidator(provider, intent, input_schema, output_schema, n_samples)],
        )
