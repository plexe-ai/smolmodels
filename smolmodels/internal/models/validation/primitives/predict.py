"""
This module defines the `PredictorValidator` class, which validates that a predictor behaves as expected.

Classes:
    - PredictorValidator: A validator class that checks the behavior of a predictor.
"""

import types
import warnings

import pandas as pd

from smolmodels.internal.common.provider import Provider
from smolmodels.internal.models.validation.validator import Validator, ValidationResult


class PredictorValidator(Validator):
    """
    A validator class that checks that a predictor behaves as expected.
    """

    def __init__(
        self,
        provider: Provider,
        intent: str,
        input_schema: dict,
        output_schema: dict,
        sample: pd.DataFrame,
    ) -> None:
        """
        Initialize the PredictorValidator with the name 'predictor'.

        :param provider: The data provider to use for generating test data.
        :param intent: The intent of the predictor.
        :param input_schema: The input schema of the predictor.
        :param output_schema: The output schema of the predictor.
        :param sample: The sample input data to test the predictor.
        """
        super().__init__("predictor")
        self.provider: Provider = provider
        self.intent: str = intent
        self.input_schema: dict = input_schema
        self.output_schema: dict = output_schema
        self.input_sample: pd.DataFrame = sample

    def validate(self, code: str) -> ValidationResult:
        """
        Validates that the given code for a predictor behaves as expected.
        :param code: prediction code to be validated
        :return: True if valid, False otherwise
        """
        try:
            predictor: types.ModuleType = self._load_predictor(code)
            self._has_predict_function(predictor)
            self._returns_output_when_called(predictor)

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
            try:
                exec(code, predictor.__dict__)
            except Exception as e:
                raise RuntimeError(f"Failed to load predictor: {str(e)}")
        return predictor

    @staticmethod
    def _has_predict_function(predictor: types.ModuleType) -> None:
        """
        Ensures that the predictor module has a valid `predict` function.
        """
        if not hasattr(predictor, "predict"):
            raise AttributeError("The module does not have a 'predict' function.")
        if not callable(predictor.predict):
            raise TypeError("'predict' is not a callable function.")

    def _returns_output_when_called(self, predictor: types.ModuleType) -> None:
        """
        Tests the `predict` function by calling it with sample inputs.
        """
        total_tests = len(self.input_sample)
        issues = []

        for i, sample in self.input_sample.iterrows():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    predictor.predict(sample.to_dict())
            except Exception as e:
                issues.append({"error": str(e), "sample": sample, "index": i})

        if len(issues) > 0:
            raise RuntimeError(f"{len(issues)}/{total_tests} calls to 'predict' failed. Issues: {issues}")
