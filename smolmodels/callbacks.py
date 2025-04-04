"""
Callbacks for model building process in SmolModels.

This module defines callback interfaces that let users hook into various stages
of the model building process, allowing for custom logging, tracking, visualization,
or other operations to be performed at key points.
"""

import logging
from abc import ABC
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type, Union

from pydantic import BaseModel

from smolmodels.constraints import Constraint
from smolmodels.internal.common.provider import Provider
from smolmodels.internal.common.utils.model_state import ModelState
from smolmodels.internal.models.entities.artifact import Artifact
from smolmodels.internal.models.entities.metric import Metric
from smolmodels.internal.models.entities.node import Node
from smolmodels.internal.models.entities.stopping_condition import StoppingCondition

logger = logging.getLogger(__name__)


@dataclass
class BuildStartInfo:
    """Information available at the start of the model building process."""

    intent: str
    """The natural language description of the model's intent."""

    identifier: str
    """Unique identifier for the model."""

    provider: Union[str, Provider]
    """The provider (LLM) used for generating the model."""

    input_schema: Optional[Type[BaseModel]] = None
    """The input schema for the model."""

    output_schema: Optional[Type[BaseModel]] = None
    """The output schema for the model."""

    constraints: List[Constraint] = field(default_factory=list)
    """Constraints the model must satisfy."""

    run_timeout: Optional[int] = None
    """Maximum time in seconds for each individual training run."""

    max_iterations: Optional[int] = None
    """Maximum number of iterations for the model building process."""

    timeout: Optional[int] = None
    """Maximum total time in seconds for the entire model building process."""


@dataclass
class BuildEndInfo(BuildStartInfo):
    """Information available at the end of the model building process."""

    state: ModelState = ModelState.READY
    """Final state of the model (READY or ERROR)."""

    metric: Optional[Metric] = None
    """Final performance metric (if successful)."""

    artifacts: List[Artifact] = field(default_factory=list)
    """Model artifacts produced (if successful)."""

    trainer_source: Optional[str] = None
    """Source code for the model trainer (if successful)."""

    predictor_source: Optional[str] = None
    """Source code for the model predictor (if successful)."""

    error: Optional[Exception] = None
    """Error that occurred (if failed)."""

    metadata: Dict[str, str] = field(default_factory=dict)
    """Additional metadata about the model."""


@dataclass
class IterationStartInfo:
    """Information available at the start of each iteration."""

    iteration: int
    """Current iteration number (0-indexed)."""

    total_iterations: Optional[int] = None
    """Total number of iterations (if known)."""

    target_metric: Optional[Metric] = None
    """Target metric to optimize."""

    stopping_condition: Optional[StoppingCondition] = None
    """The stopping condition for the search."""

    elapsed_time: float = 0.0
    """Time elapsed since the start of the search."""


@dataclass
class IterationEndInfo:
    """Information available at the end of each iteration."""

    iteration: int
    """Current iteration number (0-indexed)."""

    node: Node
    """The solution node that was evaluated in this iteration."""

    best_metric: Optional[Metric] = None
    """The best metric achieved so far across all iterations."""

    elapsed_time: float = 0.0
    """Time elapsed since the start of the search."""

    remaining_time: Optional[float] = None
    """Remaining time before timeout (if applicable)."""


class Callback(ABC):
    """
    Abstract base class for callbacks during model building.

    Callbacks allow running custom code at various stages of the model building process.
    Subclass this and implement the methods you need for your specific use case.
    """

    def on_build_start(self, info: BuildStartInfo) -> None:
        """
        Called when the model building process starts.

        :param info: Structured information about the model building process start.
        """
        pass

    def on_build_end(self, info: BuildEndInfo) -> None:
        """
        Called when the model building process ends.

        :param info: Structured information about the model building process end,
            including results or error information.
        """
        pass

    def on_iteration_start(self, info: IterationStartInfo) -> None:
        """
        Called at the start of each model building iteration.

        :param info: Structured information about the iteration start.
        """
        pass

    def on_iteration_end(self, info: IterationEndInfo) -> None:
        """
        Called at the end of each model building iteration.

        :param info: Structured information about the iteration end,
            including the evaluated node and current best metric.
        """
        pass
