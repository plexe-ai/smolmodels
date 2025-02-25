"""
This module provides file I/O utilities for saving and loading models to and from archive files.
"""

import io
import json
import logging
import pickle
import tarfile
import types
import shutil
from pathlib import Path

from smolmodels.models import Model, ModelState
from smolmodels.internal.common.utils.pydantic_utils import map_to_basemodel

logger = logging.getLogger(__name__)


def get_cache_dirs() -> tuple[Path, Path]:
    """Get the paths for model storage and extraction cache.

    Returns:
        tuple[Path, Path]: (models_dir, extract_dir)
            - models_dir: .smolcache/models/ for tar archives
            - extract_dir: .smolcache/extracted/ for cached model files
    """
    cache_root = Path(".smolcache")
    models_dir = cache_root / "models"
    extract_dir = cache_root / "extracted"

    # Ensure directories exist
    models_dir.mkdir(parents=True, exist_ok=True)
    extract_dir.mkdir(parents=True, exist_ok=True)

    return models_dir, extract_dir


def get_model_path(model: Model) -> Path:
    """Get the default path for a model archive in smolcache.

    The path will be: .smolcache/models/model-{identifier}[-{type}].tar.gz
    """
    # Remove any existing "model-" prefix from identifier
    identifier = model.identifier.replace("model-", "")
    model_name = f"model-{identifier}"
    if model.metadata.get("type"):
        model_name += f"-{model.metadata['type']}"

    models_dir, _ = get_cache_dirs()
    return models_dir / f"{model_name}.tar.gz"


def get_extract_path(model_id: str) -> Path:
    """Get the extraction cache path for a model.

    The path will be: .smolcache/extracted/{identifier}/

    Args:
        model_id: The model identifier (without 'model-' prefix)
    """
    _, extract_dir = get_cache_dirs()
    return extract_dir / model_id.replace("model-", "")


def save_model(model: Model, path: str = None) -> str:
    """
    Save a model to a single archive file in smolcache, including all components in memory.

    Archive structure:
    - metadata/
        - intent.txt
        - state.txt
        - metrics.json
        - metadata.json
    - schemas/
        - input_schema.json
        - output_schema.json
    - code/
        - trainer.py
        - predictor.py
    - artifacts/
        - [model files]

    Args:
        model: The model to save
        path: Optional custom path. If not provided, saves to smolcache/models/

    Returns:
        str: Path where the model was saved
    """
    if path is None:
        path = get_model_path(model)
    elif not isinstance(path, Path):
        path = Path(path)

    # Ensure .tar.gz extension
    if not str(path).endswith(".tar.gz"):
        path = Path(str(path).rstrip(".tar")).with_suffix(".tar.gz")

    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with tarfile.open(path, "w:gz") as tar:
            metrics_data = {}
            if model.metrics:
                metrics_data = {
                    "name": model.metrics.name,
                    "value": model.metrics.value,
                    "comparison_method": model.metrics.comparator.comparison_method.value,
                    "target": model.metrics.comparator.target,
                }

            metadata = {
                "intent": model.intent,
                "state": model.state.value,
                "metrics": metrics_data,
                "metadata": model.metadata,
                "identifier": model.identifier,
            }

            # Save each metadata item separately
            for key, value in metadata.items():
                if key in ["metrics", "metadata"]:
                    info = tarfile.TarInfo(f"metadata/{key}.json")
                    content = json.dumps(value, indent=2).encode("utf-8")
                else:
                    info = tarfile.TarInfo(f"metadata/{key}.txt")
                    content = str(value).encode("utf-8")
                info.size = len(content)
                tar.addfile(info, io.BytesIO(content))

            for name, schema in [("input_schema", model.input_schema), ("output_schema", model.output_schema)]:
                schema_dict = {name: field.annotation.__name__ for name, field in schema.model_fields.items()}
                info = tarfile.TarInfo(f"schemas/{name}.json")
                content = json.dumps(schema_dict).encode("utf-8")
                info.size = len(content)
                tar.addfile(info, io.BytesIO(content))

            if model.trainer_source:
                info = tarfile.TarInfo("code/trainer.py")
                content = model.trainer_source.encode("utf-8")
                info.size = len(content)
                tar.addfile(info, io.BytesIO(content))

            if model.predictor_source:
                info = tarfile.TarInfo("code/predictor.py")
                content = model.predictor_source.encode("utf-8")
                info.size = len(content)
                tar.addfile(info, io.BytesIO(content))

            for artifact in model.artifacts:
                artifact_path = Path(artifact)
                if artifact_path.is_dir():
                    # Recursively include all files under this directory
                    for file_path in artifact_path.rglob("*"):
                        if file_path.is_file():
                            # Build a relative path inside "artifacts/"
                            rel_path = file_path.relative_to(artifact_path.parent)
                            arcname = f"artifacts/{rel_path}"
                            info = tarfile.TarInfo(str(arcname))

                            content = file_path.read_bytes()
                            info.size = len(content)
                            tar.addfile(info, io.BytesIO(content))

                elif artifact_path.is_file():
                    # Same logic for a single file
                    arcname = f"artifacts/{artifact_path.parent.name}/{artifact_path.name}"
                    info = tarfile.TarInfo(arcname)
                    content = artifact_path.read_bytes()
                    info.size = len(content)
                    tar.addfile(info, io.BytesIO(content))
                else:
                    raise FileNotFoundError(f"Artifact not found or is not a file/directory: {artifact}")
            if model.constraints:
                info = tarfile.TarInfo("metadata/constraints.pkl")
                content = pickle.dumps(model.constraints)
                info.size = len(content)
                tar.addfile(info, io.BytesIO(content))

    except Exception as e:
        logger.error(f"Error saving model: {e}")
        if Path(path).exists():
            Path(path).unlink()
        raise

    logger.info(f"Model saved to {path}")
    return str(path)


def load_model(path_or_id: str) -> Model:
    """
    Load a model from the archive, using cached extraction if available.

    The function will:
    1. Find the model archive (by path or ID)
    2. Check if it's already extracted in the cache
    3. Extract to cache if needed
    4. Load model from cached files

    Args:
        path_or_id: Full path to tar file or model identifier (without 'model-' prefix)

    Returns:
        Model: The loaded model

    Raises:
        ValueError: If model is not found
        Exception: If there are errors during loading
    """
    if not path_or_id.endswith(".tar.gz"):
        # Remove any existing "model-" prefix
        model_id = path_or_id.replace("model-", "")
        # Look in smolcache for any file starting with this ID
        models_dir, _ = get_cache_dirs()
        matches = list(models_dir.glob(f"model-{model_id}*.tar.gz"))
        if matches:
            path = matches[0]
        else:
            path = models_dir / f"model-{model_id}.tar.gz"
    else:
        path = Path(path_or_id)
        model_id = path.stem.replace("model-", "").split("-")[0]

    if not path.exists():
        raise ValueError(f"Model not found: {path}")

    # Check if model is already extracted
    extract_path = get_extract_path(model_id)
    if not extract_path.exists():
        logger.info(f"Extracting model from {path} to {extract_path}")
        extract_path.mkdir(parents=True, exist_ok=True)
        with tarfile.open(path, "r:gz") as tar:
            tar.extractall(extract_path)
    else:
        logger.info(f"Using cached model files from {extract_path}")

    logger.info(f"Loading model from {path}")
    try:
        with tarfile.open(path, "r:gz") as tar:
            intent = tar.extractfile("metadata/intent.txt").read().decode("utf-8")
            state = ModelState(tar.extractfile("metadata/state.txt").read().decode("utf-8"))
            metrics_data = json.loads(tar.extractfile("metadata/metrics.json").read().decode("utf-8"))
            metadata = json.loads(tar.extractfile("metadata/metadata.json").read().decode("utf-8"))
            identifier = tar.extractfile("metadata/identifier.txt").read().decode("utf-8")

            # Reconstruct Metric object if metrics data exists
            from smolmodels.internal.models.entities.metric import Metric, MetricComparator, ComparisonMethod

            metrics = None
            if metrics_data:
                comparator = MetricComparator(
                    comparison_method=ComparisonMethod(metrics_data["comparison_method"]), target=metrics_data["target"]
                )
                metrics = Metric(name=metrics_data["name"], value=metrics_data["value"], comparator=comparator)

            def type_from_name(type_name: str) -> type:
                type_map = {"str": str, "int": int, "float": float, "bool": bool}
                return type_map[type_name]

            input_schema_dict = json.loads(tar.extractfile("schemas/input_schema.json").read().decode("utf-8"))
            output_schema_dict = json.loads(tar.extractfile("schemas/output_schema.json").read().decode("utf-8"))

            input_schema = map_to_basemodel(
                "InputSchema", {name: type_from_name(type_name) for name, type_name in input_schema_dict.items()}
            )
            output_schema = map_to_basemodel(
                "OutputSchema", {name: type_from_name(type_name) for name, type_name in output_schema_dict.items()}
            )

            # Load constraints if they exist
            constraints = []
            if "metadata/constraints.pkl" in [m.name for m in tar.getmembers()]:
                constraints = pickle.loads(tar.extractfile("metadata/constraints.pkl").read())
            model = Model(
                intent=intent, input_schema=input_schema, output_schema=output_schema, constraints=constraints
            )
            model.state = state
            model.metrics = metrics
            model.metadata = metadata
            model.identifier = identifier

            if "code/trainer.py" in [m.name for m in tar.getmembers()]:
                model.trainer_source = tar.extractfile("code/trainer.py").read().decode("utf-8")

            if "code/predictor.py" in [m.name for m in tar.getmembers()]:
                model.predictor_source = tar.extractfile("code/predictor.py").read().decode("utf-8")
                model.predictor = types.ModuleType("predictor")
                exec(model.predictor_source, model.predictor.__dict__)

            # Use the cached extraction directory
            model.files_path = extract_path

            # Add artifact paths from the extraction directory, including subdirectories
            for file_path in extract_path.glob("artifacts/**/*"):
                if file_path.is_file() and not file_path.name.startswith("."):
                    model.artifacts.append(str(file_path))

            logger.info(f"Model successfully loaded from {path}")
            return model

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        if extract_path.exists():
            shutil.rmtree(extract_path)
        raise
