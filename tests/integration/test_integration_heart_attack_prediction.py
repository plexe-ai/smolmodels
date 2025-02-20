import os
import pytest
from pathlib import Path
from tests.utils.utils import generate_heart_data, verify_prediction, cleanup_files
import smolmodels as sm


@pytest.fixture
def sample_data():
    """Generate synthetic heart disease data for testing"""
    return generate_heart_data(n_samples=200)


@pytest.fixture
def input_schema():
    """Define input schema - using simple types as expected by the model"""
    return {
        "age": int,
        "gender": int,
        "cp": int,
        "trtbps": int,
        "chol": int,
        "fbs": int,
        "restecg": int,
        "thalachh": int,
        "exng": int,
        "oldpeak": float,
        "slp": int,
        "caa": int,
        "thall": int,
    }


@pytest.fixture
def output_schema():
    """Define output schema - using simple types as expected by the model"""
    return {"output": int}


@pytest.fixture
def test_input():
    """Define a consistent test input for predictions"""
    return {
        "age": 61,
        "gender": 1,
        "cp": 3,
        "trtbps": 145,
        "chol": 233,
        "fbs": 1,
        "restecg": 0,
        "thalachh": 150,
        "exng": 0,
        "oldpeak": 2.3,
        "slp": 0,
        "caa": 0,
        "thall": 1,
    }


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


def test_model_with_data_and_schema(sample_data, input_schema, output_schema, test_input):
    """Test case where user provides data, input and output schema"""
    model = sm.Model(
        intent="predict the probability of heart attack based on patient features",
        input_schema=input_schema,
        output_schema=output_schema,
    )

    model.build(
        datasets=[sample_data],
        provider="openai/gpt-4o-mini",
        max_iterations=2,
        timeout=3600,
    )
    prediction = model.predict(test_input)
    verify_prediction(prediction, output_schema)
