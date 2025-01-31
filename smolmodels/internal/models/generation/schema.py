"""
Module for schema generation and handling.
"""

import logging
from typing import Dict, Tuple
from pydantic import BaseModel
import pandas as pd

from smolmodels.config import config
from smolmodels.internal.common.providers.provider import Provider

logger = logging.getLogger(__name__)


class SchemaDefinition(BaseModel):
    input_schema: Dict[str, str]
    output_schema: Dict[str, str]


def generate_schema_from_dataset(
    provider: Provider,
    intent: str,
    dataset: pd.DataFrame,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Generate input and output schemas from dataset and intent.
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
            if pd.api.types.is_bool_dtype(dataset[column]):
                types[column] = "bool"
            elif pd.api.types.is_numeric_dtype(dataset[column]):
                if pd.api.types.is_integer_dtype(dataset[column]):
                    types[column] = "int"
                else:
                    types[column] = "float"
            else:
                types[column] = "str"

        # Split into input and output schemas
        input_schema = {col: types[col] for col in dataset.columns if col != output_col}
        output_schema = {output_col: types[output_col]}

        return input_schema, output_schema

    except Exception as e:
        logger.error(f"Error inferring schema from data: {e}")
        raise


def generate_schema_from_intent(
    provider: Provider,
    intent: str,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Generate input and output schemas purely from intent using LLM."""
    try:

        response = provider.query(
            system_message=config.code_generation.prompt_schema_base.safe_substitute(),
            user_message=config.code_generation.prompt_schema_generate_from_intent.safe_substitute(intent=intent),
            # todo: change to SchemaDefinition, currently provider API doesn't like nested objects
            response_format={"type": "json_object"},
        )

        schema_definition = SchemaDefinition.model_validate_json(response)
        return schema_definition.input_schema, schema_definition.output_schema

    except Exception as e:
        logger.error(f"Error generating schema from intent: {e}")
        raise
