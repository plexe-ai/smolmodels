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
>>>        output_schema=create_model("output", **{"price": float}),
>>>        input_schema=create_model("input", **{
>>>            "bedrooms": int,
>>>            "bathrooms": int,
>>>            "square_footage": float
>>>        })
>>>    )
>>>
>>>    model.build(datasets=[pd.read_csv("houses.csv")], provider="openai:gpt-4o-mini", max_iterations=10)
>>>
>>>    prediction = model.predict({"bedrooms": 3, "bathrooms": 2, "square_footage": 1500.0})
>>>    print(prediction)
"""

import logging
import os
import uuid
from enum import Enum
from pathlib import Path
from typing import Dict, List, Type, Any

import pandas as pd
from pydantic import BaseModel

from smolmodels.config import config
from smolmodels.constraints import Constraint
from smolmodels.datasets import DatasetGenerator
from smolmodels.directives import Directive
from smolmodels.internal.common.datasets.interface import Dataset
from smolmodels.internal.common.datasets.adapter import DatasetAdapter
from smolmodels.internal.common.provider import Provider
from smolmodels.internal.common.utils.model_utils import calculate_model_size, format_code_snippet
from smolmodels.internal.common.utils.pydantic_utils import map_to_basemodel, format_schema
from smolmodels.internal.models.entities.artifact import Artifact
from smolmodels.internal.models.entities.description import (
    ModelDescription,
    SchemaInfo,
    ImplementationInfo,
    PerformanceInfo,
    CodeInfo,
)
from smolmodels.internal.models.entities.metric import Metric
from smolmodels.internal.models.generators import ModelGenerator
from smolmodels.internal.models.interfaces.predictor import Predictor
from smolmodels.internal.schemas.resolver import SchemaResolver


class ModelState(Enum):
    DRAFT = "draft"
    BUILDING = "building"
    READY = "ready"
    ERROR = "error"


logger = logging.getLogger(__name__)


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
            output_schema=create_model("output_schema", **{"price": float}),
            input_schema=create_model("input_schema", **{
                "bedrooms": int,
                "bathrooms": int,
                "square_footage": float,
            })
        )
    """

    def __init__(
        self,
        intent: str,
        input_schema: Type[BaseModel] | Dict[str, type] = None,
        output_schema: Type[BaseModel] | Dict[str, type] = None,
        constraints: List[Constraint] = None,
    ):
        """
        Initialise a model with a natural language description of its intent, as well as
        structured definitions of its input schema, output schema, and any constraints.

        :param intent: A human-readable, natural language description of the model's expected intent.
        :param input_schema: a pydantic model or dictionary defining the input schema
        :param output_schema: a pydantic model or dictionary defining the output schema
        :param constraints: A list of Constraint objects that represent rules which must be
            satisfied by every input/output pair for the model.
        """
        # todo: analyse natural language inputs and raise errors where applicable

        # The model's identity is defined by these fields
        self.intent: str = intent
        self.input_schema: Type[BaseModel] = map_to_basemodel("in", input_schema) if input_schema else None
        self.output_schema: Type[BaseModel] = map_to_basemodel("out", output_schema) if output_schema else None
        self.constraints: List[Constraint] = constraints or []
        self.training_data: Dict[str, Dataset] = dict()

        # The model's mutable state is defined by these fields
        self.state: ModelState = ModelState.DRAFT
        self.predictor: Predictor | None = None
        self.trainer_source: str | None = None
        self.predictor_source: str | None = None
        self.artifacts: List[Artifact] = []
        self.metric: Metric | None = None
        self.metadata: Dict[str, str] = dict()  # todo: initialise metadata, etc

        # Generator objects used to create schemas, datasets, and the model itself
        self.schema_resolver: SchemaResolver | None = None
        self.model_generator: ModelGenerator | None = None

        self.identifier: str = f"model-{abs(hash(self.intent))}-{str(uuid.uuid4())}"
        # Directory for any required model files
        base_dir = os.environ.get("MODEL_PATH", config.file_storage.model_cache_dir)
        self.files_path: Path = Path(base_dir) / self.identifier

    def build(
        self,
        datasets: List[pd.DataFrame | DatasetGenerator],
        provider: str = "openai/gpt-4o-mini",
        directives: List[Directive] = None,
        timeout: int = None,
        max_iterations: int = None,
        run_timeout: int = 1800,
    ) -> None:
        """
        Build the model using the provided dataset, directives, and optional data generation configuration.

        :param datasets: the datasets to use for training the model
        :param provider: the provider to use for model building
        :param directives: instructions related to the model building process - not the model itself
        :param timeout: maximum total time in seconds to spend building the model (all iterations combined)
        :param max_iterations: maximum number of iterations to spend building the model
        :param run_timeout: maximum time in seconds for each individual model training run
        :return:
        """
        # Ensure timeout, max_iterations, and run_timeout make sense
        if timeout is None and max_iterations is None:
            raise ValueError("At least one of 'timeout' or 'max_iterations' must be set")
        if run_timeout is not None and timeout is not None and run_timeout > timeout:
            raise ValueError(f"Run timeout ({run_timeout}s) cannot exceed total timeout ({timeout}s)")

        # TODO: validate that schema features are present in the dataset
        # TODO: validate that datasets do not contain duplicate features
        try:
            provider = Provider(model=provider)
            self.state = ModelState.BUILDING

            # Step 1: coerce datasets to supported formats
            self.training_data = {
                f"dataset_{i}": DatasetAdapter.coerce((data.data if isinstance(data, DatasetGenerator) else data))
                for i, data in enumerate(datasets)
            }

            # Step 2: resolve schemas
            self.schema_resolver = SchemaResolver(provider, self.intent)

            if self.input_schema is None and self.output_schema is None:
                self.input_schema, self.output_schema = self.schema_resolver.resolve(self.training_data)
            elif self.output_schema is None:
                _, self.output_schema = self.schema_resolver.resolve(self.training_data)
            elif self.input_schema is None:
                self.input_schema, _ = self.schema_resolver.resolve(self.training_data)

            # Step 3: generate model
            self.model_generator = ModelGenerator(
                self.intent, self.input_schema, self.output_schema, provider, self.constraints
            )
            generated = self.model_generator.generate(
                datasets=self.training_data,
                run_timeout=run_timeout,
                timeout=timeout,
                max_iterations=max_iterations,
            )

            # Step 4: update model state and attributes
            self.trainer_source = generated.training_source_code
            self.predictor_source = generated.inference_source_code
            self.predictor = generated.predictor
            self.artifacts = generated.model_artifacts

            # Convert Metric object to a dictionary with the entire metric object as the value
            self.metric = generated.test_performance

            # Store the model metadata from the generation process
            self.metadata.update(generated.metadata)

            # Store provider information in metadata
            self.metadata["provider"] = str(provider.model)

            self.state = ModelState.READY

        except Exception as e:
            self.state = ModelState.ERROR
            logger.error(f"Error during model building: {str(e)}")
            raise e

    def predict(self, x: Dict[str, Any], validate_input: bool = False, validate_output: bool = False) -> Dict[str, Any]:
        """
        Call the model with input x and return the output.
        :param x: input to the model
        :param validate_input: whether to validate the input against the input schema
        :param validate_output: whether to validate the output against the output schema
        :return: output of the model
        """
        if self.state != ModelState.READY:
            raise RuntimeError("The model is not ready for predictions.")
        try:
            if validate_input:
                self.input_schema.model_validate(x)
            y = self.predictor.predict(x)
            if validate_output:
                self.output_schema.model_validate(y)
            return y
        except Exception as e:
            raise RuntimeError(f"Error during prediction: {str(e)}") from e

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
        return None if self.metric is None else {self.metric.name: self.metric.value}

    def describe(self) -> ModelDescription:
        """
        Return a structured description of the model.

        :return: A ModelDescription object with various methods like to_dict(), as_text(),
                as_markdown(), to_json() for different output formats
        """
        # Create schema info
        schemas = SchemaInfo(
            input=format_schema(self.input_schema),
            output=format_schema(self.output_schema),
            constraints=[str(constraint) for constraint in self.constraints],
        )

        # Create implementation info
        implementation = ImplementationInfo(
            framework=self.metadata.get("framework", "Unknown"),
            model_type=self.metadata.get("model_type", "Unknown"),
            artifacts=[a.name for a in self.artifacts],
            size=calculate_model_size(self.artifacts),
        )

        # Create performance info
        # Convert Metric objects to string representation for JSON serialization
        metrics_dict = {}
        if hasattr(self.metric, "value") and hasattr(self.metric, "name"):  # Check if it's a Metric object
            metrics_dict[self.metric.name] = str(self.metric.value)

        performance = PerformanceInfo(
            metrics=metrics_dict,
            training_data_info={
                name: {
                    "modality": data.structure.modality,
                    "features": data.structure.features,
                    "structure": data.structure.details,
                }
                for name, data in self.training_data.items()
            },
        )

        # Create code info
        code = CodeInfo(
            training=format_code_snippet(self.trainer_source), prediction=format_code_snippet(self.predictor_source)
        )

        # Assemble and return the complete model description
        return ModelDescription(
            id=self.identifier,
            state=self.state.value,
            intent=self.intent,
            schemas=schemas,
            implementation=implementation,
            performance=performance,
            code=code,
            training_date=self.metadata.get("creation_date", "Unknown"),
            rationale=self.metadata.get("selection_rationale", "Unknown"),
            provider=self.metadata.get("provider", "Unknown"),
        )
