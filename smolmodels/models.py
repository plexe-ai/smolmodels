# smolmodels/models.py

"""
This module defines the `Model` class, which represents a machine learning model.

A `Model` is characterized by a natural language description of its intent, structured input and output schemas,
and optional constraints that the model must satisfy. This class provides methods for building the model, making
predictions, and inspecting its state, metadata, and metrics.

Key Features:
- Intent: A natural language description of the model's purpose.
- Input/Output Schema: Defines the structure and types of inputs and outputs.
- Constraints: Rules that must hold true for input/output pairs.
- Mutable State: Tracks the model's lifecycle, training metrics, and metadata.
- Build Process: Integrates solution generation with directives and callbacks.

Example:
>>>    model = Model(
>>>        intent="Given a dataset of house features, predict the house price.",
>>>        output_schema={"price": float},
>>>        input_schema={
>>>            "bedrooms": int,
>>>            "bathrooms": int,
>>>            "square_footage": float
>>>        }
>>>    )
>>>
>>>    model.build(dataset=pd.read_csv("houses.csv"), provider="openai:gpt-4o-mini")
>>>
>>>    prediction = model.predict({"bedrooms": 3, "bathrooms": 2, "square_footage": 1500.0})
>>>    print(prediction)
"""

import io
import logging
import pickle
import shutil
import tarfile
import time
import types
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Union, List, Literal, Any

import numpy as np
import pandas as pd

from smolmodels.callbacks import Callback
from smolmodels.config import config
from smolmodels.constraints import Constraint
from smolmodels.directives import Directive
from smolmodels.internal.common.providers.provider import Provider
from smolmodels.internal.common.datasets.adapter import DatasetAdapter
from smolmodels.internal.common.providers.provider_factory import ProviderFactory
from smolmodels.internal.data_generation.generator import generate_data, DataGenerationRequest
from smolmodels.internal.models.generators import generate


class ModelState(Enum):
    DRAFT = "draft"
    BUILDING = "building"
    READY = "ready"
    ERROR = "error"


logger = logging.getLogger(__name__)


@dataclass
class ModelReview:
    summary: str
    suggested_directives: List[Directive]
    # todo: this can be fleshed out further


@dataclass
class GenerationConfig:
    """Configuration for data generation/augmentation"""

    n_samples: int
    augment_existing: bool = False
    quality_threshold: float = 0.8

    def __post_init__(self):
        if self.n_samples <= 0:
            raise ValueError("Number of samples must be positive")
        if not 0 <= self.quality_threshold <= 1:
            raise ValueError("Quality threshold must be between 0 and 1")

    @classmethod
    def from_input(cls, value: Union[int, Dict[str, Any]]) -> "GenerationConfig":
        """Create config from either number or dictionary input"""
        if isinstance(value, int):
            return cls(n_samples=value)
        elif isinstance(value, dict):
            return cls(
                n_samples=value["n_samples"],
                augment_existing=value.get("augment_existing", False),
                quality_threshold=value.get("quality_threshold", 0.8),
            )
        raise ValueError(f"Invalid generate_samples value: {value}")


class Model:
    """
    Represents a model that transforms inputs to outputs according to a specified intent.

    A `Model` is defined by a human-readable description of its expected intent, as well as structured
    definitions of its input schema, output schema, and any constraints that must be satisfied by the model.

    Attributes:
        intent (str): A human-readable, natural language description of the model's expected intent.
        output_schema (dict): A mapping of output key names to their types.
        input_schema (dict): A mapping of input key names to their types.
        constraints (List[Constraint]): A list of Constraint objects that represent rules which must be
            satisfied by every input/output pair for the model.

    Example:
        model = Model(
            intent="Given a dataset of house features, predict the house price.",
            output_schema={"price": float},
            input_schema={
                "bedrooms": int,
                "bathrooms": int,
                "square_footage": float,
            }
        )
    """

    def __init__(self, intent: str, output_schema: dict, input_schema: dict, constraints: List[Constraint] = None):
        """
        Initialise a model with a natural language description of its intent, as well as
        structured definitions of its input schema, output schema, and any constraints.

        :param [str] intent: A human-readable, natural language description of the model's expected intent.
        :param [dict] output_schema: A mapping of output key names to their types.
        :param [dict] input_schema: A mapping of input key names to their types.
        :param List[Constraint] constraints: A list of Constraint objects that represent rules which must be
            satisfied by every input/output pair for the model.
        """
        # todo: analyse natural language inputs and raise errors where applicable

        # The model's identity is defined by these fields
        self.intent = intent
        self.output_schema = output_schema
        self.input_schema = input_schema
        self.constraints = constraints or []
        self.training_data: pd.DataFrame | None = None

        # The model's mutable state is defined by these fields
        self.state = ModelState.DRAFT
        self.trainer: types.ModuleType | None = None
        self.predictor: types.ModuleType | None = None
        self.trainer_source: str | None = None
        self.predictor_source: str | None = None
        self.artifacts: List[Path | str] = []
        self.metrics: Dict[str, str] = dict()
        self.metadata: Dict[str, str] = dict()  # todo: initialise metadata, etc

        # Unique identifier for the model, used in directory paths etc
        self.identifier: str = f"model-{abs(hash(self.intent))}-{str(uuid.uuid4())}"
        # Directory for any required model files
        self.files_path: Path = Path(config.file_storage.model_cache_dir) / self.identifier

    def build(
        self,
        dataset: pd.DataFrame | np.ndarray = None,
        directives: List[Directive] = None,
        generate_samples: Union[int, Dict[str, Any]] = None,
        callbacks: List[Callback] = None,
        isolation: Literal["local", "subprocess", "docker"] = "local",
        provider: str = "openai:gpt-4o-mini",
    ) -> None:
        """
        Build the model using the provided dataset, directives, and optional data generation configuration.

        :param dataset: the dataset to use for training the model
        :param directives: instructions related to the model building process - not the model itself
        :param generate_samples: synthetic data generation configuration
        :param callbacks: functions that are called during the model building process
        :param isolation: level of isolation under which model build should be executed
        :param provider: the provider to use for model building
        :return:
        """
        try:
            # Attempt to select an LLM provider based on input
            provider: Provider = ProviderFactory.create(provider)
            self.state = ModelState.BUILDING

            # Convert dataset to internal format
            self.training_data = DatasetAdapter.convert(dataset) if dataset is not None else None

            # Handle data generation if requested
            if generate_samples is not None:
                datagen_config = GenerationConfig.from_input(generate_samples)

                request = DataGenerationRequest(
                    intent=self.intent,
                    input_schema=self.input_schema,
                    output_schema=self.output_schema,
                    n_samples=datagen_config.n_samples,
                    augment_existing=datagen_config.augment_existing,
                    quality_threshold=datagen_config.quality_threshold,
                    existing_data=self.training_data,
                )

                synthetic_data = generate_data(provider, request)

                # Handle augmentation
                if self.training_data is not None and datagen_config.augment_existing:
                    self.training_data = pd.concat([self.training_data, synthetic_data], ignore_index=True)
                else:
                    self.training_data = synthetic_data

            # Validate we have training data from some source
            if self.training_data is None:
                raise ValueError("No training data available. Provide dataset or generate_samples.")

            # Generate the model
            generated = generate(
                intent=self.intent,
                input_schema=self.input_schema,
                output_schema=self.output_schema,
                dataset=self.training_data,
                provider=provider,
                filedir=self.files_path,
                constraints=self.constraints,
                directives=directives,
                callbacks=callbacks,
                isolation=isolation,
            )
            self.trainer, self.trainer_source, self.predictor, self.predictor_source, self.artifacts, self.metrics = (
                generated
            )

            self.state = ModelState.READY
            print("✅ Model built successfully.")
        except Exception as e:
            self.state = ModelState.ERROR
            logger.error(f"Error during model building: {str(e)}")
            raise e

    def predict(self, x: Any) -> Any:
        """
        Call the model with input x and return the output.
        :param x: input to the model
        :return: output of the model
        """
        if self.state != ModelState.READY:
            raise RuntimeError("The model is not ready for predictions.")
        return self.predictor.predict(x)

    def get_state(self) -> ModelState:
        """
        Return the current state of the model.
        :return: the current state of the model
        """
        return self.state

    def get_metadata(self) -> dict:
        """
        Return metadata about the model.
        :return: metadata about the model
        """
        return self.metadata

    def get_metrics(self) -> dict:
        """
        Return metrics about the model.
        :return: metrics about the model
        """
        return self.metrics

    def describe(self) -> dict:
        """
        Return a human-readable description of the model.
        :return: a human-readable description of the model
        """
        return {
            "intent": self.intent,
            "output_schema": self.output_schema,
            "input_schema": self.input_schema,
            "constraints": [str(constraint) for constraint in self.constraints],
            "state": self.state,
            "metadata": self.metadata,
            "metrics": self.metrics,
        }

    def review(self) -> ModelReview:
        """
        Return a review of the model, which is a structured object consisting of a natural language
        summary, suggested directives to apply, and more.
        :return: a review of the model
        """
        raise NotImplementedError("Review functionality is not yet implemented.")


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
            if model.trainer:
                trainer_info = io.BytesIO(model.trainer_source.encode("utf-8"))
                trainer_tarinfo = tarfile.TarInfo(name="trainer.py")
                trainer_tarinfo.size = len(model.trainer_source)
                tar.addfile(trainer_tarinfo, trainer_info)

            # Save the predictor source code
            if model.predictor:
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
                "output_schema": model.output_schema,
                "input_schema": model.input_schema,
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
            output_schema=model_data["output_schema"],
            input_schema=model_data["input_schema"],
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
                shutil.copy(artifact_path, model.files_path / artifact_path.name)
                model.artifacts.append(str(model.files_path / artifact_path.name))

        with open(trainer_path, "r") as f:
            model.trainer = types.ModuleType("trainer")
            model.trainer_source = f.read()
            exec(model.trainer_source, model.trainer.__dict__)

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
