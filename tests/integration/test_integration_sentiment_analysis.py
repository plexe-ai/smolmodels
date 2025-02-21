import os
from pathlib import Path

import pandas as pd
import pytest
from pydantic import create_model

import smolmodels as sm
from tests.utils.utils import verify_prediction, cleanup_files


@pytest.fixture
def sentiment_data():
    """Generate sample sentiment data for testing"""
    positive_texts = [
        "This product exceeded my expectations!",
        "Great service and amazing quality",
        "I absolutely love this product",
        "Best purchase I've ever made",
        "Fantastic experience overall",
    ]
    negative_texts = [
        "Very disappointed with the quality",
        "Would not recommend this",
        "Poor service and slow delivery",
        "Product broke after first use",
        "Waste of money",
    ]

    data = []
    for text in positive_texts:
        data.append({"text": text, "sentiment": "positive"})
    for text in negative_texts:
        data.append({"text": text, "sentiment": "negative"})

    return pd.DataFrame(data)


@pytest.fixture
def model_dir(tmpdir):
    """Create and manage a temporary directory for model files"""
    model_path = Path(tmpdir) / "models"
    model_path.mkdir(exist_ok=True)
    return model_path


@pytest.fixture(autouse=True)
def run_around_tests(model_dir):
    cleanup_files(model_dir)
    model_dir.mkdir(exist_ok=True)
    os.environ["MODEL_PATH"] = str(model_dir)
    yield
    # Teardown
    cleanup_files(model_dir)


def test_sentiment_classification(sentiment_data):
    """Test sentiment classification model end-to-end"""
    model = sm.Model(
        intent="Classify text sentiment into positive or negative categories.",
        input_schema=create_model("in", **{"text": "str"}),
        output_schema=create_model("out", **{"sentiment": "str"}),
    )

    dataset = sm.DatasetGenerator(
        description="A dataset of text reviews and their corresponding sentiment labels.",
        provider="openai/gpt-4o-mini",
        schema=create_model("data", **{"text": "str", "sentiment": "str"}),
        data=sentiment_data,
    )
    dataset.generate(10)

    # Build the model
    model.build(
        datasets=[dataset],
        max_iterations=3,  # Keep iterations low for testing
        timeout=360,  # Keep timeout low for testing
    )

    # Test predictions
    test_cases = [
        {"input": {"text": "This product exceeded my expectations!"}, "expected_schema": {"sentiment": "str"}},
        {"input": {"text": "Very disappointing experience, I hate it"}, "expected_schema": {"sentiment": "str"}},
    ]

    for test_case in test_cases:
        prediction = model.predict(test_case["input"])
        # Verify prediction schema
        verify_prediction(prediction, test_case["expected_schema"])
        # Verify prediction format
        assert isinstance(prediction, dict)
        assert "sentiment" in prediction
        assert prediction["sentiment"] in ["positive", "negative"]

    # Verify model description
    description = model.describe()
    assert isinstance(description, dict)
    assert "intent" in description
    assert "input_schema" in description
    assert "output_schema" in description
