"""
This module contains the Pydantic schemas for the data generation service's API requests and responses.
"""

from typing import List

from pydantic import BaseModel


# base entities
class DatasetSchema(BaseModel):
    """A dataset to be generated or validated."""

    column_names: List[str]
    column_types: List[str]
    column_descriptors: List[str]
    column_nullable: List[bool]


# data generation requests and responses
class DataGenerationTask(BaseModel):
    """A request to generate a dataset"""

    problem_description: str
    n_output_records: int
    dataset_schema: DatasetSchema = None
    seed_data_path: str = None
    output_data_path: str = None


class DataGenerationResponse(BaseModel):
    """A response to a data generation request"""

    data_output_path: str


# data validation requests and responses
class DataValidationTask(BaseModel):
    """A request to validate a dataset"""

    synthetic_data_path: str
    report_output_path: str
    real_data_path: str = None
    data_schema: DatasetSchema = None


class DataValidationResponse(BaseModel):
    """A response to a data validation request"""

    report_output_path: str
