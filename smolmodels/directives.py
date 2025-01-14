# smolmodels.directives.py
"""
This module defines the `Directive` class, which represents technical instructions that guide the model
generation process in the `smolmodels` library.

Directives consist of:
- A natural language instruction that provides specific guidance during model generation.
- An optional condition that determines when the directive is applicable, evaluated based on the
  current state of the model-building process.

Example:
    directive = Directive(
        directive="Optimize for memory usage.",
        condition=lambda state: state.get("resources", {}).get("GPU", False)
    )

    if directive.is_applicable(state):
        print(directive)
"""

from typing import Callable, Any


class Directive:
    """
    Represents a directive for the model generation process.

    A `Directive` consists of:
    1. A natural language instruction to guide the model generation.
    2. An optional condition that determines whether the directive is applicable based on the
       current state of the model-building process.

    Attributes:
        directive (str): The natural language instruction.
        condition (Callable[[Any], bool]): A callable that evaluates to True if the directive
            should be applied given the current state. Defaults to always True.
    """

    def __init__(self, directive: str, condition: Callable[[Any], bool] = None):
        """
        Initialize a Directive.

        :param directive: A natural language instruction.
        :param condition: A callable that evaluates to True or False based on the state. Defaults to always True.
        """
        self.directive = directive
        self.condition = None

        # todo: implement the handling of conditions - the main question is what should be the
        # input of the condition function? The current state of the model, something else?
        if condition is not None:
            raise NotImplementedError("Condition is not implemented yet.")

    def is_applicable(self, state: Any) -> bool:
        """
        Determine if the directive is applicable given the current state.

        :param state: The current state of the model-building process.
        :return: True if the directive is applicable, False otherwise.
        """
        # todo: once the condition handling is implemented, use it here
        return True

    def __str__(self):
        return f"Directive: {self.directive}"
