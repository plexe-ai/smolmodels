# smolmodels/internal/models/validation/predictor.py

"""
This module defines the `PredictorValidator` class, which validates that a predictor behaves as expected.

Classes:
    - PredictorValidator: A validator class that checks the behavior of a predictor.
"""

import types
from smolmodels.internal.models.validation.validator import Validator, ValidationResult


class PredictorValidator(Validator):
    """
    A validator class that checks that a predictor behaves as expected.
    """

    def __init__(self, sample_input: dict):
        """
        Initialize the PredictorValidator with the name 'predictor'.
        """
        super().__init__("predictor")
        self.sample_input = sample_input

    def validate(self, code: str) -> ValidationResult:
        """
        Validates that the given code for a predictor behaves as expected.
        :param code: prediction code to be validated
        :return: True if valid, False otherwise
        """
        try:
            # compile the inference code into a module
            predictor: types.ModuleType = types.ModuleType("test_predictor")
            exec(code, predictor.__dict__)
            # check if the module has a 'predict' function
            assert hasattr(predictor, "predict"), "The module does not have a 'predict' function."
            # check if the 'predict' function is callable
            assert callable(predictor.predict), "'predict' is not a callable function."
            # todo: try calling the 'predict' function
            # output = predictor.predict(self.sample_input)
            # assert output is not None, "The 'predict' function returned None."
        except Exception as e:
            return ValidationResult(
                self.name,
                False,
                message=f"Prediction code is not valid: {str(e)}.",
                exception=e,
            )
