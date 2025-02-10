# internal/models/validation/validators.py

from smolmodels.internal.models.validation.validator import Validator, ValidationResult
from smolmodels.internal.models.validation.syntax import SyntaxValidator
from smolmodels.internal.models.validation.predictor import PredictorValidator
from smolmodels.internal.common.provider import Provider


class TrainingCodeValidator(Validator):
    def __init__(self):
        """
        Initialize the TrainingValidator with the name 'training'.
        """
        super().__init__("training")
        self.syntax_validator = SyntaxValidator()

    def validate(self, code: str) -> ValidationResult:
        """
        Validates that the given code for a training behaves as expected.
        :param code: training code to be validated
        :return: True if valid, False otherwise
        """
        return self.syntax_validator.validate(code)


class PredictionCodeValidator(Validator):
    def __init__(
        self, provider: Provider, intent: str, input_schema: dict, output_schema: dict, n_samples: int = 10
    ) -> None:
        """
        Initialize the PredictionValidator with the name 'prediction'.
        """
        super().__init__("prediction")
        self.syntax_validator = SyntaxValidator()
        self.predictor_validator = PredictorValidator(provider, intent, input_schema, output_schema, n_samples)

    def validate(self, code: str) -> ValidationResult:
        """
        Validates that the given code for a prediction behaves as expected.
        :param code: prediction code to be validated
        :return: True if valid, False otherwise
        """
        r = self.syntax_validator.validate(code)
        if not r.passed:
            return r
        return self.predictor_validator.validate(code)
