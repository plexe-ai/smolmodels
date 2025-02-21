"""
This module provides file I/O utilities for saving and loading models to and from archive files.
"""

import io
import logging
import pickle
import shutil
import tarfile
import time
import types
from pathlib import Path

from smolmodels.config import config
from smolmodels.models import Model, ModelState
from smolmodels.internal.common.utils.pydantic_utils import create_model_from_fields

logger = logging.getLogger(__name__)


def save_model(model: Model, path: str) -> None:
    """
    Save a model to a single archive file, including trainer, predictor, and artifacts.

    :param model: the model to save
    :param path: the path to save the model to
    """
    if not path.endswith(".tar.gz"):
        path += ".tar.gz"
    try:
        with tarfile.open(path, "w:gz") as tar:
            # Save the trainer source code
            if model.trainer_source:
                trainer_info = io.BytesIO(model.trainer_source.encode("utf-8"))
                trainer_tarinfo = tarfile.TarInfo(name="trainer.py")
                trainer_tarinfo.size = len(model.trainer_source)
                tar.addfile(trainer_tarinfo, trainer_info)

            # Save the predictor source code
            if model.predictor_source:
                predictor_info = io.BytesIO(model.predictor_source.encode("utf-8"))
                predictor_tarinfo = tarfile.TarInfo(name="predictor.py")
                predictor_tarinfo.size = len(model.predictor_source)
                tar.addfile(predictor_tarinfo, predictor_info)

            # Collect and save all artifacts
            for artifact in model.artifacts:
                artifact_path = Path(artifact)
                if artifact_path.exists():
                    tar.add(artifact_path, arcname=artifact_path.name)
                else:
                    raise FileNotFoundError(f"Artifact not found: {artifact}")

            # Save the model metadata
            model_data = {
                "intent": model.intent,
                "output_schema": model.output_schema.model_fields,
                "input_schema": model.input_schema.model_fields,
                "constraints": model.constraints,
                "metrics": model.metrics,
                "metadata": model.metadata,
                "state": model.state.value,
                "identifier": model.identifier,
            }

            model_data_bytes = io.BytesIO(pickle.dumps(model_data))
            model_data_tarinfo = tarfile.TarInfo(name="model_data.pkl")
            model_data_tarinfo.size = model_data_bytes.getbuffer().nbytes
            tar.addfile(model_data_tarinfo, model_data_bytes)

    except Exception as e:
        logger.error(f"Error saving model, cleaning up tarfile: {e}")
        if Path(path).exists():
            Path(path).unlink()
        raise e
    finally:
        # Cleanup model cache directory
        if model.files_path.exists():
            shutil.rmtree(model.files_path)


# todo: move to a separate module
def load_model(path: str) -> Model:
    """
    Load a model from the archive created by `save_model`.
    :param path: the path to load the model from
    :return: the loaded model
    """
    import tarfile

    try:
        # Ensure smolmodels cache directory exists
        cache_dir: Path = Path(config.file_storage.model_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Create a temporary directory to extract the archive
        temp_dir: Path = cache_dir / f"loading-{time.time()}"
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Extract the archive
        with tarfile.open(path, "r:gz") as tar:
            tar.extractall(temp_dir)

        # Load model data
        model_data_path = temp_dir / "model_data.pkl"
        with open(model_data_path, "rb") as f:
            model_data = pickle.load(f)

        # Create the model instance
        model = Model(
            intent=model_data["intent"],
            input_schema=create_model_from_fields("InputSchema", model_data["input_schema"]),
            output_schema=create_model_from_fields("OutputSchema", model_data["output_schema"]),
            constraints=model_data["constraints"],
        )

        print(model.intent)
        print(model.files_path)

        model.identifier = model_data["identifier"]
        model.files_path = model.files_path.parent / model.identifier

        # Restore state, metrics, and metadata
        model.state = ModelState(model_data["state"])
        model.metrics = model_data["metrics"]
        model.metadata = model_data["metadata"]

        # Ensure model cache directory exists
        model.files_path.mkdir(parents=True, exist_ok=True)

        # Copy trainer and predictor source code to the model's cache directory
        shutil.copy(temp_dir / "trainer.py", model.files_path / "trainer.py")
        shutil.copy(temp_dir / "predictor.py", model.files_path / "predictor.py")

        trainer_path = model.files_path / "trainer.py"
        predictor_path = model.files_path / "predictor.py"

        # Restore artifacts
        if temp_dir.exists():
            for artifact_path in temp_dir.iterdir():
                if artifact_path.is_file():
                    shutil.copy(artifact_path, model.files_path / artifact_path.name)
                    model.artifacts.append(str(model.files_path / artifact_path.name))
                elif artifact_path.is_dir():
                    shutil.copytree(artifact_path, model.files_path / artifact_path.name)
                    model.artifacts.append(str(model.files_path / artifact_path.name))

        with open(trainer_path, "r") as f:
            model.trainer_source = f.read()

        with open(predictor_path, "r") as f:
            model.predictor = types.ModuleType("predictor")
            model.predictor_source = f.read()
            exec(model.predictor_source, model.predictor.__dict__)

        logger.info(f"Model successfully loaded from {path}.")
        return model

    except Exception as e:
        logger.error(f"Error loading model, cleaning up model files: {e}")
        if model is not None and model.files_path.exists():
            shutil.rmtree(model.files_path)
        raise e

    finally:
        # Cleanup temp directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
