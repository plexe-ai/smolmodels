# smolmodels/internal/models/validation/predictor.py

"""
This module defines the `PredictorValidator` class, which validates that a predictor behaves as expected.

Classes:
    - PredictorValidator: A validator class that checks the behavior of a predictor.
"""

import warnings
import types
import random
import hypothesis.strategies as st

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
        self.output_schema = output_schema
        self.input_strategy = st.fixed_dictionaries({k: st.from_type(v) for k, v in input_schema.items()})

    def validate(self, code: str) -> ValidationResult:
        """
        Validates that the given code for a predictor behaves as expected.
        :param code: prediction code to be validated
        :return: True if valid, False otherwise
        """
        try:
            predictor = self._load_predictor(code)
            self._validate_predictor_structure(predictor)
            self._test_predict_function(predictor)

            return ValidationResult(self.name, True, "Prediction code is valid.")

        except Exception as e:
            return ValidationResult(
                self.name,
                False,
                message=f"Prediction code is not valid: {str(e)}.",
                exception=e,
            )

    @staticmethod
    def _load_predictor(code: str) -> types.ModuleType:
        """
        Compiles and loads the predictor module from the given code.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            predictor = types.ModuleType("test_predictor")
            exec(code, predictor.__dict__)
        return predictor

    @staticmethod
    def _validate_predictor_structure(predictor: types.ModuleType) -> None:
        """
        Ensures that the predictor module has a valid `predict` function.
        """
        if not hasattr(predictor, "predict"):
            raise AttributeError("The module does not have a 'predict' function.")
        if not callable(predictor.predict):
            raise TypeError("'predict' is not a callable function.")

    def _test_predict_function(self, predictor) -> None:
        """
        Tests the `predict` function by calling it with sample inputs.
        """
        issues = []
        total_tests = 100

        for _ in range(total_tests):
            sample_input = self.input_strategy.example()
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    predictor.predict(sample_input)
            except Exception as e:
                issues.append(f"Error: {str(e)} | Input: {sample_input}")

        failed_tests = len(issues)

        if failed_tests == total_tests:
            raise RuntimeError(
                f"All {total_tests} calls to 'predict' failed. Sample issues: {random.sample(issues, 5)}"
            )
        if failed_tests >= total_tests * 0.5:
            raise RuntimeError(
                f"{failed_tests}/{total_tests} calls to 'predict' failed. Sample issues: {random.sample(issues, 5)}"
            )
