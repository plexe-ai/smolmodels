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

Example Usage:
    model = Model(
        intent="Given a dataset of house features, predict the house price.",
        output_schema={"price": float},
        input_schema={
            "bedrooms": int,
            "bathrooms": int,
            "square_footage": float
        }
    )

    model.build(dataset="houses.csv", directives=[Directive("Optimize for memory usage")])

    prediction = model.predict({"bedrooms": 3, "bathrooms": 2, "square_footage": 1500.0})
    print(prediction)

"""
import types
import pandas as pd
import logging
import pickle
from enum import Enum
from typing import Dict, Optional, Union, List, Literal, Any
from dataclasses import dataclass
from pathlib import Path

from smolmodels.callbacks import Callback
from smolmodels.constraints import Constraint
from smolmodels.directives import Directive
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
        self.training_data = None
        self.synthetic_data = None

        # The model's mutable state is defined by these fields
        self.state = ModelState.DRAFT
        self.trainer: types.ModuleType | None = None
        self.predictor: types.ModuleType | None = None
        self.artifacts: List[Path | str] = []
        self.metrics: Dict[str, str] = dict()
        self.metadata: Dict[str, str] = dict()

        # todo: metrics should be chosen based on problem, model-type, etc.
        # todo: initialise metadata, etc
        logger.debug(f"Model initialised with state: {vars(self)}")

    def build(
        self,
        dataset: Optional[Union[str, pd.DataFrame]] = None,
        directives: List[Directive] = None,
        generate_samples: Optional[Union[int, Dict[str, Any]]] = None,
        callbacks: List[Callback] = None,
        isolation: Literal["local", "subprocess", "docker"] = "local",
    ) -> None:
        try:
            self.state = ModelState.BUILDING

            # Handle existing dataset
            if isinstance(dataset, str):
                self.training_data = pd.read_csv(dataset)
            elif isinstance(dataset, pd.DataFrame):
                self.training_data = dataset.copy()

            # Handle data generation if requested
            if generate_samples is not None:
                config = GenerationConfig.from_input(generate_samples)

                request = DataGenerationRequest(
                    intent=self.intent,
                    input_schema=self.input_schema,
                    output_schema=self.output_schema,
                    n_samples=config.n_samples,
                    augment_existing=config.augment_existing,
                    quality_threshold=config.quality_threshold,
                    existing_data=self.training_data,
                )

                self.synthetic_data = generate_data(request)

                # Handle augmentation
                if self.training_data is not None and config.augment_existing:
                    self.training_data = pd.concat([self.training_data, self.synthetic_data], ignore_index=True)
                else:
                    self.training_data = self.synthetic_data

            # Validate we have training data from some source
            if self.training_data is None:
                raise ValueError("No training data available. Provide dataset or generate_samples.")

            # Generate the model
            generated = generate(
                self.intent,
                self.input_schema,
                self.output_schema,
                self.training_data,
                self.constraints,
                directives,
                callbacks,
                isolation,
            )
            self.trainer, self.predictor, self.artifacts, self.metrics = generated

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
        # todo: this is a placeholder, implement the actual model prediction logic
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
    Save a model to a file.
    :param model: the model to save
    :param path: the path to save the model to
    """
    # Ensure the path has extension
    if not path.endswith(".pmb"):
        path += ".pmb"

    try:
        with open(path, "wb") as file:
            pickle.dump(model, file)
        logger.info(f"Model successfully saved to {path}.")
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        raise e


def load_model(path: str) -> Model:
    """
    Load a model from a file.
    :param path: the path to load the model from
    :return: the loaded model
    """
    try:
        with open(path, "rb") as file:
            model = pickle.load(file)
        logger.info(f"Model successfully loaded from {path}.")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise e
