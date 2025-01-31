"""
Module for schema generation and handling.
"""

import logging
from typing import Dict, Optional, Tuple
import pandas as pd

from smolmodels.config import config
from smolmodels.internal.common.providers.provider import Provider

logger = logging.getLogger(__name__)


def generate_schema(
    provider: Provider,
    intent: str,
    dataset: pd.DataFrame,  # No longer optional since we always have data
    input_schema: Optional[Dict[str, str]] = None,
    output_schema: Optional[Dict[str, str]] = None,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Generate input and output schemas using dataset and intent.
    Uses dataset column names and types, with LLM only identifying the target column.
    """
    try:
        # Let LLM identify which column should be output
        columns_str = "\n".join(f"- {col}" for col in dataset.columns)
        output_col = provider.query(
            system_message=config.code_generation.prompt_schema_base.safe_substitute(),
            user_message=config.code_generation.prompt_schema_identify_target.safe_substitute(
                columns=columns_str, intent=intent
            ),
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
