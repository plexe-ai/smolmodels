import io
import sys
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from smolmodels.internal.common.provider import Provider
from smolmodels.internal.data_generation.generator import DataGenerationRequest, generate_data


def test_data_generation_progress_bars():
    """Test that progress bars are displayed during data generation"""
    # Capture stdout to verify progress bar output
    captured_output = io.StringIO()
    sys.stdout = captured_output

    # Mock provider and its query method
    mock_provider = MagicMock(spec=Provider)
    mock_provider.query.return_value = '[{"col1": 1, "col2": "a"}]'

    # Create a test request
    request = DataGenerationRequest(
        intent="Test data generation",
        input_schema={"col1": int},
        output_schema={"col2": str},
        n_samples=10
    )

    # Generate data
    with patch('pandas.read_csv') as mock_read_csv:
        mock_read_csv.return_value = pd.DataFrame({"col1": [1], "col2": ["a"]})
        generate_data(mock_provider, request)

    # Get captured output
    output = captured_output.getvalue()

    # Verify progress bar elements are present
    assert "Total samples generated" in output
    assert "Generating batches" in output
    assert "samples/s" in output
    assert "%" in output

    # Reset stdout
    sys.stdout = sys.__stdout__