# tests/test_data_generation.py

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
from smolmodels import Model
from smolmodels.internal.data_generation.core.generation.utils.oversampling import oversample_with_smote


@pytest.fixture
def sample_schema():
    """Test schema for house price prediction"""
    return {"input_schema": {"square_feet": float, "bedrooms": int, "location": str}, "output_schema": {"price": float}}


@pytest.fixture
def mock_successful_generation():
    """Mock successful data generation response - now returns larger batches"""

    def generate(*args, **kwargs):
        batch_size = kwargs.get("batch_size", 50)
        return pd.DataFrame(
            {
                "square_feet": np.random.uniform(1000, 3000, batch_size),
                "bedrooms": np.random.randint(2, 6, batch_size),
                "location": np.random.choice(["suburban", "urban", "rural"], batch_size),
                "price": np.random.uniform(200000, 600000, batch_size),
            }
        )

    return generate


class TestDataGeneration:
    """Test suite for data generation with mocked responses"""

    def test_basic_generation(self, sample_schema, mock_successful_generation):
        """Test basic data generation"""
        with patch(
            "smolmodels.internal.data_generation.core.generation.combined.CombinedDataGenerator._generate_batch",
            side_effect=mock_successful_generation,
        ):
            model = Model(intent="Predict house prices based on features", **sample_schema)
            model.build(generate_samples=50)

            assert model.training_data is not None
            assert len(model.training_data) >= 45  # Allow for small variations
            assert all(col in model.training_data.columns for col in ["square_feet", "bedrooms", "location", "price"])

    def test_data_augmentation(self, sample_schema, mock_successful_generation):
        """Test data augmentation with existing dataset"""
        existing_data = pd.DataFrame(
            {
                "square_feet": [1000, 1500, 2000],
                "bedrooms": [2, 3, 4],
                "location": ["A", "B", "C"],
                "price": [200000, 300000, 400000],
            }
        )

        with patch(
            "smolmodels.internal.data_generation.core.generation.combined.CombinedDataGenerator._generate_batch",
            side_effect=mock_successful_generation,
        ):
            model = Model(intent="Predict house prices based on features", **sample_schema)
            model.build(dataset=existing_data.copy(), generate_samples={"n_samples": 7, "augment_existing": True})
            assert len(model.training_data) >= 8  # 3 original + at least 5 new
            assert len(model.synthetic_data) >= 5
            # Verify original data is preserved
            original_subset = existing_data.sort_values("square_feet").reset_index(drop=True)
            generated_subset = (
                model.training_data[model.training_data["square_feet"].isin(existing_data["square_feet"])]
                .sort_values("square_feet")
                .reset_index(drop=True)
            )
            pd.testing.assert_frame_equal(original_subset, generated_subset, check_dtype=False)

    def test_handles_empty_response(self, sample_schema):
        """Test handling of empty response from generator"""
        with patch(
            "smolmodels.internal.data_generation.core.generation.combined.CombinedDataGenerator._generate_batch",
            return_value=pd.DataFrame(),
        ):
            model = Model(intent="Predict house prices based on features", **sample_schema)
            with pytest.raises(RuntimeError, match="Failed to generate any valid data"):
                model.build(generate_samples=500)

    def test_handles_api_error(self, sample_schema):
        """Test handling of API errors"""
        with patch(
            "smolmodels.internal.data_generation.core.generation.combined.CombinedDataGenerator._generate_batch",
            side_effect=Exception("API Error"),
        ):
            model = Model(intent="Predict house prices based on features", **sample_schema)
            with pytest.raises(RuntimeError):
                model.build(generate_samples=5)


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
