"""
This module provides utility functions for manipulating Pydantic models.
"""

from pydantic import BaseModel, create_model
from typing import Type, List


def merge_models(model_name: str, models: List[Type[BaseModel]]) -> Type[BaseModel]:
    """
    Merge multiple Pydantic models into a single model. The ordering of the list determines
    the overriding precedence of the models; the last model in the list will override any fields
    with the same name in the preceding models.

    :param model_name: The name of the new model to create.
    :param models: A list of Pydantic models to merge.
    :return: A new Pydantic model that combines the input models.
    """
    fields = dict()
    for model in models:
        for name, properties in model.model_fields.items():
            fields[name] = (properties.annotation, ... if properties.is_required() else properties.default)
    return create_model(model_name, **fields)


def create_model_from_fields(model_name: str, model_fields: dict) -> Type[BaseModel]:
    """
    Create a Pydantic model from a dictionary of fields.

    :param model_name: The name of the model to create.
    :param model_fields: A dictionary of field names to field properties.
    """
    for name, properties in model_fields.items():
        model_fields[name] = (properties.annotation, ... if properties.is_required() else properties.default)
    return create_model(model_name, **model_fields)
