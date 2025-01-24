import abc
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """
    Represents the result of a validation.

    Attributes:
        name (str): The name of the validation.
        passed (bool): Whether the validation passed or not.
        message (str, optional): A message providing details about the validation result.
        exception (Exception, optional): An exception that was raised during validation, if any.
    """

    name: str
    passed: bool
    message: str = None
    exception: Exception | None = None


class Validator(abc.ABC):
    """
    Abstract base class for validators.

    Attributes:
        name (str): The name of the validator.
    """

    @abc.abstractmethod
    def __init__(self, name: str):
        """
        Initializes the validator with a name.

        :param [str] name: The name of the validator.
        """
        self.name = name

    @abc.abstractmethod
    def validate(self, code: str) -> ValidationResult:
        """
        Validates the given code.

        :param [str] code: The code to validate.
        :return: [ValidationResult] The result of the validation.
        """
        pass
