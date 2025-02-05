# smolmodels/internal/models/validation/predictor.py

"""
This module defines the `PredictorValidator` class, which validates that a predictor behaves as expected.

Classes:
    - PredictorValidator: A validator class that checks the behavior of a predictor.
"""

import warnings
import hypothesis.strategies as st
import types

from hypothesis.errors import NonInteractiveExampleWarning

from smolmodels.internal.models.validation.validator import Validator, ValidationResult


warnings.filterwarnings("ignore", category=NonInteractiveExampleWarning)


class PredictorValidator(Validator):
    """
    A validator class that checks that a predictor behaves as expected.
    """

    def __init__(self, input_schema: dict, output_schema: dict) -> None:
        """
        Initialize the PredictorValidator with the name 'predictor'.
        """
        super().__init__("predictor")
        self.input_schema: dict = input_schema
        self.output_schema: dict = output_schema

    def validate(self, code: str) -> ValidationResult:
        """
        Validates that the given code for a predictor behaves as expected.
        :param code: prediction code to be validated
        :return: True if valid, False otherwise
        """
        try:
            # Compile the inference code into a module
            predictor: types.ModuleType = types.ModuleType("test_predictor")
            exec(code, predictor.__dict__)
            # Check module has a 'predict' function
            assert hasattr(predictor, "predict"), "The module does not have a 'predict' function."
            # Check 'predict' function is callable
            assert callable(predictor.predict), "'predict' is not a callable function."
            # Check 'predict' function works with sample input
            input_strategy = st.fixed_dictionaries({k: st.from_type(v) for k, v in self.input_schema.items()})
            for _ in range(20):
                sample_input = input_strategy.example()
                try:
                    predictor.predict(sample_input)
                except Exception as e:
                    raise RuntimeError(f"Error calling 'predict' function with sample input: {str(e)}") from e
            # If all checks were passed, return a valid result
            return ValidationResult(self.name, True, "Prediction code is valid.")
        except Exception as e:
            # If any check failed, return an invalid result
            return ValidationResult(
                self.name,
                False,
                message=f"Prediction code is not valid: {str(e)}.",
                exception=e,
            )
