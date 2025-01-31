"""
Module for schema generation and handling.
"""

import logging
import json
from typing import Dict, Optional, Any, Tuple
import pandas as pd
import numpy as np

from smolmodels.internal.common.providers.provider import Provider

logger = logging.getLogger(__name__)


def generate_schema(
    provider: Provider,
    intent: str,
    dataset: Optional[Any] = None,
    input_schema: Optional[Dict[str, str]] = None,
    output_schema: Optional[Dict[str, str]] = None,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Generate input and output schemas using intent and available data.
    When dataset is provided, use its column names instead of generating new ones.
    """
    # If dataset is available, use its column names
    if dataset is not None and isinstance(dataset, pd.DataFrame):
        # Let LLM identify which column should be output
        data_preview = "\n".join(f"- {col}" for col in dataset.columns)
        prompt = f"""
        Given these columns from the dataset:
        {data_preview}
        
        For this ML task: {intent}
        
        Which column is the target/output variable? Return ONLY the exact column name, nothing else.
        The column name must be exactly as shown above.
        """

        try:
            output_col = provider.query(
                system_message="You are an expert ML engineer identifying target variables.", user_message=prompt
            ).strip()

            # Verify output column exists
            if output_col not in dataset.columns:
                logger.warning(f"LLM suggested non-existent column {output_col}, defaulting to last column")
                output_col = dataset.columns[-1]

            # Determine types for all columns
            types = {}
            for column in dataset.columns:
                if pd.api.types.is_numeric_dtype(dataset[column]):
                    if pd.api.types.is_integer_dtype(dataset[column]):
                        types[column] = "int"
                    else:
                        types[column] = "float"
                elif pd.api.types.is_bool_dtype(dataset[column]):
                    types[column] = "bool"
                else:
                    types[column] = "str"

            # Split into input and output schemas
            input_schema_inferred = {col: types[col] for col in dataset.columns if col != output_col}
            output_schema_inferred = {output_col: types[output_col]}

            # Use provided schemas if available, otherwise use inferred
            final_input = input_schema or input_schema_inferred
            final_output = output_schema or output_schema_inferred

            return _validate_schemas(final_input, final_output)

        except Exception as e:
            logger.error(f"Error inferring schema from data: {e}")
            raise

    if dataset is not None:
        try:
            if isinstance(dataset, pd.DataFrame):
                columns_str = "\n".join(f"- {col}: {dataset[col].dtype}" for col in dataset.columns)
                data_context = f"\nAvailable columns in the dataset:\n{columns_str}"
            elif isinstance(dataset, np.ndarray):
                data_context = f"\nNumpy array with shape {dataset.shape} and dtype {dataset.dtype}"
        except Exception as e:
            logger.warning(f"Failed to generate data context: {e}")

    # Generate schemas using LLM
    system_message = """You are an expert ML engineer designing data schemas.
You MUST return your response as a valid JSON object with exactly these fields:
{
    "input_schema": {"field_name": "type", ...},
    "output_schema": {"field_name": "type", ...}
}
Types MUST be one of: "int", "float", "str", "bool"
Do not include any explanation or additional text."""

    prompt = f"""Based on this machine learning task, generate appropriate input and output schemas.
The schemas should define the semantic meaning of each field.

Task description: {intent}
{data_context}

Output schema should contain what needs to be predicted.
Input schema should contain features used for prediction.

Respond ONLY with a JSON object in this exact format:
{{
    "input_schema": {{"field_name": "type", ...}},
    "output_schema": {{"field_name": "type", ...}}
}}"""

    try:
        response_text = provider.query(system_message=system_message, user_message=prompt).strip()

        try:
            response = json.loads(response_text)
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON response from LLM: {response_text}")
            raise ValueError("Failed to parse schema generation response")

        if not isinstance(response, dict) or "input_schema" not in response or "output_schema" not in response:
            raise ValueError("Schema generation response missing required fields")

        generated_input = response["input_schema"]
        generated_output = response["output_schema"]

        if not isinstance(generated_input, dict) or not isinstance(generated_output, dict):
            raise ValueError("Generated schemas must be dictionaries")

        # Use provided schemas if available, otherwise use generated
        final_input = input_schema or generated_input
        final_output = output_schema or generated_output

        return _validate_schemas(final_input, final_output)

    except Exception as e:
        logger.error(f"Schema generation failed: {str(e)}")
        raise ValueError(f"Failed to generate schema: {str(e)}")


def _validate_schemas(
    input_schema: Dict[str, str], output_schema: Dict[str, str]
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Validate input and output schemas."""
    # Validate types
    valid_types = {"int", "float", "str", "bool"}
    for schema in [input_schema, output_schema]:
        for name, type_str in schema.items():
            if type_str not in valid_types:
                raise ValueError(f"Invalid type {type_str} for {name}. Must be one of {valid_types}")

    # Check for overlaps
    input_fields = set(input_schema.keys())
    output_fields = set(output_schema.keys())
    if overlap := (input_fields & output_fields):
        raise ValueError(f"Fields cannot be both input and output: {overlap}")

    # Ensure we have at least one field in each
    if not input_fields:
        raise ValueError("Must have at least one input field")
    if not output_fields:
        raise ValueError("Must have at least one output field")

    return input_schema, output_schema
