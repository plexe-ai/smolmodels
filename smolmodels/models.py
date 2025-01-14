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

from enum import Enum
from typing import Union, List, Generator, Literal, Any
from dataclasses import dataclass

from smolmodels.callbacks import Callback
from smolmodels.constraints import Constraint
from smolmodels.directives import Directive


class ModelState(Enum):
    DRAFT = "draft"
    BUILDING = "building"
    READY = "ready"
    ERROR = "error"


@dataclass
class ModelReview:
    summary: str
    suggested_directives: List[Directive]
    # todo: this can be fleshed out further


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

        # The model's mutable state is defined by these fields
        # todo: this is WIP, trying to flesh out what the model's internal state might look like
        self.state = ModelState.DRAFT
        self.trainer = None  # todo: this is the object that was used to train the model, but does it need to exist?
        self.predictor = None  # todo: this is an object that loads the model and makes predictions
        self.metrics = dict()  # todo: this is a dictionary of metrics that the model has achieved
        self.metadata = dict()  # todo: this is a dictionary of metadata about the model
        # todo: metrics should be chosen based on problem, model-type, etc.
        # todo: initialise metadata, etc

    def build(
        self,
        dataset: Union[str, Generator],
        directives: List[Directive] = None,
        callbacks: List[Callback] = None,
        isolation: Literal["local", "subprocess", "docker"] = "local",
    ) -> None:
        # todo: implement properly, this is a placeholder
        raise NotImplementedError("Generation of the model is not yet implemented.")

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
