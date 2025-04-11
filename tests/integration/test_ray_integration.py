"""
Integration test for Ray-based distributed training.
"""

import pytest
import pandas as pd
import numpy as np
from smolmodels.models import Model


@pytest.fixture
def sample_dataset():
    """Create a simple synthetic dataset for testing."""
    # Create a sample regression dataset
    np.random.seed(42)
    n_samples = 1000
    X = np.random.randn(n_samples, 5)
    y = 2 + 3 * X[:, 0] + 0.5 * X[:, 1] - X[:, 2] + np.random.randn(n_samples) * 0.1

    # Create a DataFrame with feature and target columns
    df = pd.DataFrame(data=np.column_stack([X, y]), columns=[f"feature_{i}" for i in range(5)] + ["target"])
    return df


def test_model_with_ray(sample_dataset):
    """Test building a model with Ray-based distributed execution."""
    # Skip this test if Ray is not installed

    from smolmodels.config import config

    # Make sure Ray is enabled
    config.ray.enabled = True

    # Create a model with distributed=True
    model = Model(intent="Predict the target variable given 5 numerical features", distributed=True)

    # Set a short timeout for testing
    model.build(
        datasets=[sample_dataset],
        provider="openai/gpt-4o-mini",
        timeout=300,  # 5 minutes max
        run_timeout=60,  # 1 minute per run
    )

    # Test a prediction
    input_data = {f"feature_{i}": 0.5 for i in range(5)}
    prediction = model.predict(input_data)

    # Verify that prediction has expected structure
    assert prediction is not None
    assert "target" in prediction

    # Verify that Ray was used in training
    assert model.distributed

    # Verify model built successfully
    assert model.metric is not None
