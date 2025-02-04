import os
import logging
import tempfile
import pytest
from pathlib import Path
from tests.utils.utils import generate_heart_data

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def test_dir():
    """Create a temporary directory for test files"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


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


def cleanup_files(model_dir=None):
    """Clean up any files created during tests"""
    files_to_clean = [
        "smolmodels.log",
        "heart_attack_model.pmb",
    ]
    # Clean up files in current directory
    for file in files_to_clean:
        try:
            if os.path.exists(file):
                os.remove(file)
        except Exception as e:
            logger.warning(f"Failed to clean up {file}: {e}")

    # Clean up files in model directory
    if model_dir is not None and model_dir.exists():
        for file in model_dir.glob("*"):
            try:
                file.unlink()
            except Exception as e:
                logger.warning(f"Failed to clean up {file}: {e}")


@pytest.fixture(autouse=True)
def run_around_tests(model_dir):
    cleanup_files(model_dir)
    model_dir.mkdir(exist_ok=True)
    os.environ["MODEL_PATH"] = str(model_dir)
    yield
    # Teardown
    cleanup_files(model_dir)


def verify_prediction(prediction, expected_schema=None):
    """Verify that a prediction matches expected format"""
    assert isinstance(prediction, dict), "Prediction should be a dictionary"
    assert len(prediction) > 0, "Prediction should not be empty"

    if expected_schema:
        assert set(prediction.keys()) == set(
            expected_schema.keys()
        ), f"Prediction keys {prediction.keys()} don't match schema keys {expected_schema.keys()}"

    output_value = list(prediction.values())[0]
    assert isinstance(output_value, (int, float)), f"Prediction value should be numeric, got {type(output_value)}"
    assert 0 <= float(output_value) <= 1 or output_value in [
        0,
        1,
    ], f"Prediction value should be between 0 and 1 or binary, got {output_value}"


def test_model_with_data_and_schema(sample_data, input_schema, output_schema, test_input):
    """Test case where user provides data, input and output schema"""
    import smolmodels as sm

    logger.info("Starting test_model_with_data_and_schema")
    logger.debug(f"Sample data shape: {sample_data.shape}")
    logger.debug(f"Input schema keys: {list(input_schema.keys())}")
    logger.debug(f"Output schema keys: {list(output_schema.keys())}")

    model = sm.Model(
        intent="predict the probability of heart attack based on patient features",
        input_schema=input_schema,
        output_schema=output_schema,
    )

    model.build(dataset=sample_data)
    prediction = model.predict(test_input)
    verify_prediction(prediction, output_schema)


def test_model_with_data_and_generate(sample_data, input_schema, output_schema, test_input):
    """Test case where user provides data and generate_samples"""
    import smolmodels as sm

    logger.info("Starting test_model_with_data_and_generate")

    model = sm.Model(
        intent="predict the probability of heart attack based on patient features",
        input_schema=input_schema,
        output_schema=output_schema,
    )

    model.build(dataset=sample_data, generate_samples=10)
    prediction = model.predict(test_input)
    verify_prediction(prediction, output_schema)
