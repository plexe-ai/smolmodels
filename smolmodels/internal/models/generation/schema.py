"""
Module for schema inference and generation, supporting both data-driven and LLM-based approaches.
"""

import logging
import json
from typing import Dict, Tuple
import pandas as pd

from smolmodels.internal.common.providers.openai import OpenAIProvider

logger = logging.getLogger(__name__)


def infer_schema_from_data(data: pd.DataFrame, intent: str = None) -> Tuple[Dict, Dict]:
    """
    Infer input and output schemas from a DataFrame and optional intent.

    Args:
        data: Input DataFrame
        intent: Optional natural language description to help identify outputs

    Returns:
        Tuple of (input_schema, output_schema)
    """
    # If we have intent, use it to identify likely output columns
    if intent:
        output_cols = identify_output_columns(data.columns, intent)
    else:
        # Default to last column as output if no intent provided
        output_cols = [data.columns[-1]]

    # Determine types for each column
    all_types = {}
    for column in data.columns:
        if pd.api.types.is_numeric_dtype(data[column]):
            if pd.api.types.is_integer_dtype(data[column]):
                all_types[column] = "int"
            else:
                all_types[column] = "float"
        elif pd.api.types.is_bool_dtype(data[column]):
            all_types[column] = "bool"
        else:
            all_types[column] = "str"

    # Split into input and output schemas
    input_schema = {col: all_types[col] for col in data.columns if col not in output_cols}
    output_schema = {col: all_types[col] for col in output_cols}

    return input_schema, output_schema


def generate_schemas_from_intent(intent: str) -> Tuple[Dict, Dict]:
    """
    Generate input and output schemas purely from intent using LLM.

    Args:
        intent: Natural language description of the ML task

    Returns:
        Tuple of (input_schema, output_schema)
    """
    client = OpenAIProvider()

    prompt = f"""
    Analyze this machine learning task and generate appropriate input and output schemas.
    Include only essential features directly mentioned or strongly implied by the task.
    
    Task description: {intent}
    
    Respond with a JSON object:
    {{
        "input_schema": {{"feature_name": "type", ...}},
        "output_schema": {{"target_name": "type", ...}}
    }}
    
    Use only these types: "int", "float", "str", "bool".
    The output_schema should contain what needs to be predicted.
    The input_schema should contain the features used for prediction.
    """

    try:
        response = json.loads(
            client.query(system_message="You are an expert ML engineer designing data schemas.", user_message=prompt)
        )

        return response["input_schema"], response["output_schema"]

    except Exception as e:
        logger.error(f"Failed to generate schema from intent: {str(e)}")
        raise


def identify_output_columns(columns, intent: str) -> list:
    """
    Identify which columns are likely outputs based on intent.

    Args:
        columns: List of column names
        intent: Natural language description of task

    Returns:
        List of column names identified as outputs
    """
    client = OpenAIProvider()

    prompt = f"""
    Given these columns: {', '.join(columns)}
    And this ML task: {intent}
    
    Which column(s) should be predicted? Return only the column name(s) separated by commas, nothing else.
    """

    try:
        response = client.query(
            system_message="You are an expert ML engineer identifying target variables.", user_message=prompt
        ).strip()

        # Split response into list and verify columns exist
        output_cols = [col.strip() for col in response.split(",")]
        valid_cols = [col for col in output_cols if col in columns]

        if not valid_cols:
            # Fallback to last column if no valid outputs identified
            return [columns[-1]]

        return valid_cols

    except Exception as e:
        logger.error(f"Failed to identify output columns: {str(e)}")
        # Fallback to last column
        return [columns[-1]]
