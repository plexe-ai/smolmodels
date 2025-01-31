"""Unit tests for schema generation functionality."""

import pytest
import pandas as pd
from unittest.mock import Mock

from smolmodels.internal.models.generation.schema import generate_schema


@pytest.fixture
def mock_provider():
    provider = Mock()
    # Mock the provider to return 'target' as the output column
    provider.query.return_value = "target"
    return provider


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "feature1": [1, 2, 3],
            "feature2": [1.5, 2.5, 3.5],
            "text_col": ["a", "b", "c"],
            "bool_col": [True, False, True],
            "target": [0, 1, 0],
        }
    )


def test_basic_schema_generation(mock_provider, sample_df):
    """Test basic schema generation from DataFrame."""
    input_schema, output_schema = generate_schema(provider=mock_provider, intent="predict target", dataset=sample_df)

    # Check input schema types are correctly inferred
    assert input_schema["feature1"] == "int"
    assert input_schema["feature2"] == "float"
    assert input_schema["text_col"] == "str"
    assert input_schema["bool_col"] == "bool"

    # Check output schema
    assert output_schema == {"target": "int"}

    # Verify LLM was called with correct prompt
    mock_provider.query.assert_called_once()


def test_respects_existing_schemas(mock_provider, sample_df):
    """Test that existing schemas are respected."""
    input_schema = {"feature1": "float"}  # Override inferred type
    output_schema = {"target": "float"}  # Override inferred type

    final_input, final_output = generate_schema(
        provider=mock_provider,
        intent="predict target",
        dataset=sample_df,
        input_schema=input_schema,
        output_schema=output_schema,
    )

    # Check that provided schemas are preserved
    assert final_input == input_schema
    assert final_output == output_schema


def test_schema_validation_error():
    """Test schema validation with invalid types."""
    input_schema = {"feature1": "invalid_type"}
    output_schema = {"target": "int"}

    with pytest.raises(ValueError, match="Invalid type invalid_type"):
        generate_schema(
            provider=Mock(),
            intent="predict target",
            dataset=pd.DataFrame({"feature1": [1], "target": [0]}),
            input_schema=input_schema,
            output_schema=output_schema,
        )


def test_target_column_fallback(mock_provider, sample_df):
    """Test fallback to last column when LLM suggests invalid column."""
    # Make LLM suggest a non-existent column
    mock_provider.query.return_value = "non_existent_column"

    input_schema, output_schema = generate_schema(provider=mock_provider, intent="predict target", dataset=sample_df)

    # Should fall back to last column (target)
    assert "target" in output_schema
    assert len(output_schema) == 1
