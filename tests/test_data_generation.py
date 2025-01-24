# tests/test_data_generation.py

import os
import tempfile
import numpy as np
import pytest
import pandas as pd
from smolmodels import Model
from smolmodels.internal.data_generation.core.generation.utils.oversampling import oversample_with_smote


@pytest.fixture
def house_price_model():
    """Fixture for creating a standard house price model"""
    return Model(
        intent="Predict house prices based on features",
        input_schema={"square_feet": float, "bedrooms": int, "location": str},
        output_schema={"price": float},
    )


@pytest.fixture
def sample_house_data():
    """Fixture for sample housing data"""
    return pd.DataFrame(
        {
            "square_feet": [1000, 1500, 2000],
            "bedrooms": [2, 3, 4],
            "location": ["A", "B", "C"],
            "price": [200000, 300000, 400000],
        }
    )


def test_basic_generation(house_price_model):
    """Test basic case with just number of samples"""
    model = house_price_model

    # Generate synthetic data only
    model.build(generate_samples=10)

    # Verify generated data
    assert model.training_data is not None
    assert len(model.training_data) == 10
    assert all(col in model.training_data.columns for col in ["square_feet", "bedrooms", "location", "price"])

    # Check data types
    assert model.training_data["square_feet"].dtype.kind in ["f", "i"]  # float or int
    assert model.training_data["bedrooms"].dtype.kind == "i"  # int
    assert model.training_data["location"].dtype.kind in ["O", "S"]  # object or string
    assert model.training_data["price"].dtype.kind in ["f", "i"]  # float or int


def test_augmentation(house_price_model, sample_house_data):
    """Test augmenting existing dataset"""
    model = house_price_model

    # Generate additional samples
    model.build(
        dataset=sample_house_data.copy(),  # Use copy to avoid modifying original
        generate_samples={"n_samples": 10, "augment_existing": True},
    )

    # Verify combined data
    assert len(model.training_data) == 13  # 3 original + 10 synthetic
    assert len(model.synthetic_data) == 10  # Only synthetic portion

    # Verify original data is preserved
    original_data = sample_house_data.sort_values(by=["square_feet"]).reset_index(drop=True)

    # Find exact matches using all columns from original data
    mask = pd.Series(True, index=model.training_data.index)
    for col in original_data.columns:
        mask &= model.training_data[col].isin(original_data[col])

    combined_original = model.training_data[mask].sort_values(by=["square_feet"]).reset_index(drop=True)

    pd.testing.assert_frame_equal(original_data, combined_original, check_dtype=False)


def test_advanced_generation(house_price_model):
    """Test generation with advanced configuration"""
    model = house_price_model

    # Use advanced configuration
    model.build(generate_samples={"n_samples": 10, "quality_threshold": 0.9})

    assert model.training_data is not None
    assert len(model.training_data) == 10


def test_generate_with_file(house_price_model, sample_house_data):
    """Test generation using data from a file"""
    model = house_price_model

    # Save sample data to temporary file
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        sample_house_data.to_csv(tmp.name, index=False)

    try:
        # Generate additional samples using file
        model.build(dataset=tmp.name, generate_samples={"n_samples": 10, "augment_existing": True})

        assert len(model.training_data) == 13  # 3 original + 100 synthetic
        assert len(model.synthetic_data) == 10

    finally:
        # Cleanup
        if os.path.exists(tmp.name):
            os.remove(tmp.name)


def test_invalid_inputs(house_price_model):
    """Test error handling for invalid inputs"""
    model = house_price_model

    # Test missing data source
    with pytest.raises(ValueError, match="No training data available"):
        model.build()

    # Test invalid generate_samples type
    with pytest.raises(ValueError, match="Invalid generate_samples value"):
        model.build(generate_samples="invalid")

    # Test negative samples
    with pytest.raises(ValueError, match="Number of samples must be positive"):
        model.build(generate_samples={"n_samples": -100})


class TestSMOTEOversampling:
    def test_smote_maintains_class_proportions(self):
        """Test that SMOTE maintains relative class frequencies when oversampling"""
        df = pd.DataFrame(
            {"feature1": range(20), "feature2": range(20), "target": [0] * 14 + [1] * 6}  # 70% class 0, 30% class 1
        )

        n_samples = len(df) * 2
        oversampled_df = oversample_with_smote(df, "target", n_samples)

        original_proportions = df["target"].value_counts(normalize=True)
        new_proportions = oversampled_df["target"].value_counts(normalize=True)

        pd.testing.assert_series_equal(
            original_proportions.sort_index(), new_proportions.sort_index(), check_exact=False, rtol=0.1
        )
        assert len(oversampled_df) == n_samples

    def test_smote_with_multiple_classes(self):
        """Test SMOTE works correctly with more than two classes"""
        df = pd.DataFrame(
            {
                "feature1": range(24),
                "feature2": range(24),
                "target": [0] * 6 + [1] * 6 + [2] * 6 + [3] * 6,  # 4 classes, 6 samples each
            }
        )

        n_samples = 48  # Double the size
        oversampled_df = oversample_with_smote(df, "target", n_samples)

        original_proportions = df["target"].value_counts(normalize=True)
        new_proportions = oversampled_df["target"].value_counts(normalize=True)

        pd.testing.assert_series_equal(
            original_proportions.sort_index(), new_proportions.sort_index(), check_exact=False, rtol=0.1
        )
        assert len(oversampled_df) == n_samples

    def test_smote_with_extreme_imbalance(self):
        """Test SMOTE handles extremely imbalanced datasets"""
        df = pd.DataFrame(
            {
                "feature1": range(106),
                "feature2": range(106),
                "target": [0] * 100 + [1] * 6,  # ~94% vs ~6%, with enough samples for SMOTE
            }
        )

        n_samples = 200
        oversampled_df = oversample_with_smote(df, "target", n_samples)

        original_proportions = df["target"].value_counts(normalize=True)
        new_proportions = oversampled_df["target"].value_counts(normalize=True)

        pd.testing.assert_series_equal(
            original_proportions.sort_index(), new_proportions.sort_index(), check_exact=False, rtol=0.1
        )
        # Allow for ±1 sample difference
        assert abs(len(oversampled_df) - n_samples) <= 1

    def test_smote_preserves_feature_distributions(self):
        """Test SMOTE generates synthetic samples that preserve feature distributions"""
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, 120),
                "feature2": np.random.uniform(-1, 1, 120),
                "target": [0] * 80 + [1] * 40,
            }
        )

        n_samples = 240  # Double the size
        oversampled_df = oversample_with_smote(df, "target", n_samples)

        for feature in ["feature1", "feature2"]:
            assert np.abs(df[feature].mean() - oversampled_df[feature].mean()) < 0.5
            assert np.abs(df[feature].std() - oversampled_df[feature].std()) < 0.5
