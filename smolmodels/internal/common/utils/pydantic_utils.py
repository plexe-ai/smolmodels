"""
This module provides utility functions for manipulating Pydantic models.
"""

from pydantic import BaseModel, create_model, TypeAdapter
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


def map_to_basemodel(name: str, schema: dict | Type[BaseModel]) -> Type[BaseModel]:
    """
    Ensure that the schema is a Pydantic model or a dictionary, and return the model.

    :param [str] name: the name to be given to the model class
    :param [dict] schema: the schema to be converted to a Pydantic model
    :return: the Pydantic model
    """
    # Dictionary: convert to Pydantic model, if possible
    if isinstance(schema, dict):
        try:
            TypeAdapter(dict[str, type]).validate_python(schema)
            return create_model(name, **schema)
        except Exception as e:
            raise ValueError(f"Invalid schema definition: {e}")
    # Pydantic model: return as is
    elif isinstance(schema, type) and issubclass(schema, BaseModel):
        return schema
    # All other schema types are invalid
    else:
        raise TypeError("Schema must be a Pydantic model or a dictionary.")
