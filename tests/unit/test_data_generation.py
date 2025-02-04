# tests/test_data_generation.py

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from smolmodels import Model
from smolmodels.internal.common.provider import Provider
from smolmodels.internal.data_generation.core.generation.utils.oversampling import oversample_with_smote
from smolmodels.internal.data_generation.generator import DataGenerationRequest
from smolmodels.internal.models.generators import GenerationResult


@pytest.fixture
def sample_schema():
    """Test schema for house price prediction"""
    return {
        "input_schema": {"square_feet": "float", "bedrooms": "int", "location": "str"},
        "output_schema": {"price": "float"},
    }


@pytest.fixture
def mock_generated_data():
    """Mock data generation output"""
    return pd.DataFrame(
        {
            "square_feet": np.random.uniform(1000, 3000, 50),
            "bedrooms": np.random.randint(2, 6, 50),
            "location": np.random.choice(["suburban", "urban", "rural"], 50),
            "price": np.random.uniform(200000, 600000, 50),
        }
    )


class TestDataGeneration:
    """Test suite for data generation with comprehensive mocking"""

    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        """Setup all required mocks for the test class"""
        # Mock the data generation function
        self.mock_generate_data = patch("smolmodels.models.generate_data", return_value=pd.DataFrame()).start()

        # Mock the model generation function
        self.mock_generate = patch(
            "smolmodels.models.ModelGenerator.generate", return_value=(GenerationResult("", "", MagicMock(), [], None))
        ).start()

        yield

        # Stop all mocks after the test
        patch.stopall()

    def test_basic_generation(self, sample_schema, mock_generated_data):
        """Test basic data generation"""
        self.mock_generate_data.return_value = mock_generated_data

        model = Model(
            intent="Predict house prices based on features",
            input_schema={"square_feet": int, "bedrooms": int},
            output_schema={"price": float},
        )
        model.build(generate_samples=50)

        # Verify generate_data was called with correct parameters
        self.mock_generate_data.assert_called_once()
        call_args = self.mock_generate_data.call_args[0]
        assert isinstance(call_args[0], Provider)
        assert isinstance(call_args[1], DataGenerationRequest)
        assert call_args[1].n_samples == 50
        assert not call_args[1].augment_existing
        assert call_args[1].existing_data is None

        # Verify model was built with the generated data
        assert model.training_data is not None
        assert len(model.training_data) == 50
        assert model.state.value == "ready"

    def test_data_augmentation(self, sample_schema, mock_generated_data):
        """Test data augmentation with existing dataset"""
        self.mock_generate_data.return_value = mock_generated_data

        existing_data = pd.DataFrame(
            {
                "square_feet": [1000, 1500, 2000],
                "bedrooms": [2, 3, 4],
                "location": ["A", "B", "C"],
                "price": [200000, 300000, 400000],
            }
        )

        model = Model(intent="Predict house prices based on features", **sample_schema)
        model.build(dataset=existing_data.copy(), generate_samples={"n_samples": 7, "augment_existing": True})

        # Verify generate_data was called with correct parameters
        self.mock_generate_data.assert_called_once()
        call_args = self.mock_generate_data.call_args[0]
        assert isinstance(call_args[0], Provider)
        assert isinstance(call_args[1], DataGenerationRequest)
        assert call_args[1].n_samples == 7
        assert call_args[1].augment_existing
        pd.testing.assert_frame_equal(call_args[1].existing_data, existing_data)

        # Verify final dataset includes both original and synthetic data
        assert len(model.training_data) > len(existing_data)
        assert model.state.value == "ready"

    def test_handles_no_data(self, sample_schema):
        """Test handling when no data is provided"""
        model = Model(intent="Predict house prices based on features", **sample_schema)

        with pytest.raises(ValueError, match="No data available. Provide dataset or generate_samples."):
            model.build()  # No dataset or generate_samples provided
            assert model.state.value == "error"

    def test_handles_generation_error(self, sample_schema):
        """Test handling of generation errors"""
        self.mock_generate_data.side_effect = Exception("Generation failed")

        model = Model(intent="Predict house prices based on features", **sample_schema)
        with pytest.raises(Exception, match="Generation failed"):
            model.build(generate_samples=5)

        assert model.state.value == "error"

    def test_dataset_only(self, sample_schema):
        """Test building model with only a dataset (no generation)"""
        existing_data = pd.DataFrame(
            {
                "square_feet": [1000, 1500, 2000],
                "bedrooms": [2, 3, 4],
                "location": ["A", "B", "C"],
                "price": [200000, 300000, 400000],
            }
        )

        model = Model(intent="Predict house prices based on features", **sample_schema)
        model.build(dataset=existing_data.copy())

        # Verify generate_data was not called
        self.mock_generate_data.assert_not_called()

        # Verify the model was built with the provided dataset
        pd.testing.assert_frame_equal(model.training_data, existing_data)
        assert model.state.value == "ready"


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
