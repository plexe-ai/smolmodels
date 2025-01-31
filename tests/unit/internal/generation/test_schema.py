"""Unit tests for schema generation functionality."""

import pytest
import pandas as pd
from unittest.mock import Mock

from smolmodels.internal.models.generation.schema import generate_schema_from_dataset
from smolmodels.models import Model


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
    input_schema, output_schema = generate_schema_from_dataset(
        provider=mock_provider, intent="predict target", dataset=sample_df
    )

    # Check input schema types are correctly inferred
    assert input_schema["feature1"] == "int"
    assert input_schema["feature2"] == "float"
    assert input_schema["text_col"] == "str"
    assert input_schema["bool_col"] == "bool"

    # Check output schema
    assert output_schema == {"target": "int"}

    # Verify LLM was called with correct prompt
    mock_provider.query.assert_called_once()


def test_target_column_fallback(mock_provider, sample_df):
    """Test fallback to last column when LLM suggests invalid column."""
    # Make LLM suggest a non-existent column
    mock_provider.query.return_value = "non_existent_column"

    input_schema, output_schema = generate_schema_from_dataset(
        provider=mock_provider, intent="predict target", dataset=sample_df
    )

    # Should fall back to last column (target)
    assert "target" in output_schema
    assert len(output_schema) == 1


def test_schema_in_describe():
    """Test that describe() correctly includes schemas."""
    model = Model(intent="predict target", input_schema={"age": int, "height": float}, output_schema={"result": bool})

    description = model.describe()

    assert "input_schema" in description
    assert description["input_schema"] == {"age": int, "height": float}
    assert "output_schema" in description
    assert description["output_schema"] == {"result": bool}
