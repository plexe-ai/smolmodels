"""
This module provides utility functions for manipulating Pydantic models.
"""

from pydantic import BaseModel, create_model
from typing import Type, List


def merge_models(models: List[Type[BaseModel]]) -> Type[BaseModel]:
    """
    Merge multiple Pydantic models into a single model. The ordering of the list determines
    the overriding precedence of the models; the last model in the list will override any fields
    with the same name in the preceding models.

    :param models: A list of Pydantic models to merge.
    :return: A new Pydantic model that combines the input models.
    """
    fields = dict()
    for model in models:
        for name, properties in model.model_fields.items():
            fields[name] = (properties.annotation, ... if properties.is_required() else properties.default)
    return create_model("MergedModel", **fields)
