"""
Module: smolmodels/internal/models/validation/syntax

This module defines the SyntaxValidator class, which is responsible for validating the syntax
of Python code using the AST module.

Classes:
    - SyntaxValidator: A validator class that checks the syntax of Python code.
"""

import ast

from smolmodels.internal.models.validation.validator import Validator, ValidationResult


class SyntaxValidator(Validator):
    """
    A validator class that checks the syntax of Python code using the AST module.
    """

    def __init__(self):
        """
        Initialize the SyntaxValidator with the name 'syntax'.
        """
        super().__init__("syntax")

    def validate(self, code: str) -> ValidationResult:
        """
        Validate the generated code using the Python AST module.

        :param [str] code: The Python code to be validated.
        :return: [ValidationResult] The result of the validation, indicating whether the syntax is valid.
        """
        try:
            ast.parse(code)
            return ValidationResult(self.name, True)
        except SyntaxError as e:
            return ValidationResult(self.name, False, message="Syntax is not valid.", exception=e)
