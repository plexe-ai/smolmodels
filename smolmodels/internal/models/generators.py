"""
This module provides the main function `generate` for generating machine learning models based on
a given problem statement, input schema, and output schema. The function explores the solution space,
generates training and inference code, and returns callable functions for training and prediction.
"""

import logging
import os
import shutil
import time
import types
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd

from smolmodels.config import config
from smolmodels.constraints import Constraint
from smolmodels.directives import Directive
from smolmodels.internal.common.provider import Provider
from smolmodels.internal.models.entities.graph import Graph
from smolmodels.internal.models.entities.metric import Metric
from smolmodels.internal.models.entities.node import Node
from smolmodels.internal.models.entities.stopping_condition import StoppingCondition
from smolmodels.internal.models.execution.process_executor import ProcessExecutor
from smolmodels.internal.models.generation.inference import InferenceCodeGenerator
from smolmodels.internal.models.generation.planning import SolutionPlanGenerator
from smolmodels.internal.models.generation.training import TrainingCodeGenerator
from smolmodels.internal.models.search.best_first_policy import BestFirstSearchPolicy
from smolmodels.internal.models.search.policy import SearchPolicy
from smolmodels.internal.models.utils import join_task_statement, execute_node
from smolmodels.internal.models.validation.composites.training import TrainingCodeValidator
from smolmodels.internal.models.validation.composites.inference import InferenceCodeValidator

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    training_source_code: str
    inference_source_code: str
    inference_module: types.ModuleType
    model_artifacts: List[Path]
    performance: Metric


class ModelGenerator:
    """
    Encapsulates the process of generating machine learning models based on a given problem statement.
    The model generator sets up the solution graph, code generators, and other dependencies required to
    explore solution options. It generates training and inference code, and returns a callable predictor.

    Attributes:
        intent: The intent of the model to generate.
        input_schema: The input schema for the model.
        output_schema: The output schema for the model.
        provider: The provider to use for generating models.
        filedir: The directory to store model artifacts.
        constraints: A list of constraints to apply to the model.
        graph: The solution graph for the model generation process.
        plan_generator: The solution plan generator for the model generation process.
        train_generator: The training code generator for the model generation process.
        infer_generator: The inference code generator for the model generation process.
        search_policy: The search policy for the model generation process.
        train_validators: A list of validators for training code.
        infer_validators: A list of validators for inference code.

    Example:
    >>> from smolmodels.internal.models.generators import ModelGenerator
    >>> ...
    >>> generator = ModelGenerator(
    >>>     intent="classify",
    >>>     input_schema={"age": int},
    >>>     output_schema={"label": str},
    >>>     provider=Provider(),
    >>>     filedir=Path("./models"),
    >>> )
    """

    def __init__(
        self,
        intent: str,
        input_schema: dict,
        output_schema: dict,
        provider: Provider,
        filedir: Path,
        constraints: List[Constraint] = None,
    ) -> None:
        """
        Initialises the model generator with the given problem statement, input schema, and output schema.

        :param intent: The intent of the model to generate.
        :param input_schema: The input schema for the model.
        :param output_schema: The output schema for the model.
        :param provider: The provider to use for generating models.
        :param filedir: The directory to store model artifacts.
        :param constraints: A list of constraints to apply to the model.
        """
        # Set up the basic configuration of the model generator
        self.intent: str = intent
        self.input_schema: dict = input_schema
        self.output_schema: dict = output_schema
        self.constraints: List[Constraint] = constraints or []
        self.provider: Provider = provider
        self.filedir: Path = filedir
        self.isolation: str = "subprocess"  # todo: parameterise and support other isolation methods
        # Initialise the model solution graph, code generators, etc.
        self.graph: Graph = Graph()
        self.plan_generator = SolutionPlanGenerator(provider)  # todo: allow dependency injection for these
        self.train_generator = TrainingCodeGenerator(provider)
        self.infer_generator = InferenceCodeGenerator(provider)
        self.search_policy: SearchPolicy = BestFirstSearchPolicy(self.graph)
        self.train_validators: TrainingCodeValidator = TrainingCodeValidator()

    def generate(
        self,
        dataset: pd.DataFrame,
        timeout: int = None,
        max_iterations=None,
        directives: List[Directive] = None,
    ) -> GenerationResult:
        """
        Generates a machine learning model based on the given problem statement, input schema, and output schema.

        :param dataset: The dataset to use for training the model.
        :param timeout: The maximum time to spend generating the model, in seconds.
        :param max_iterations: The maximum number of iterations to spend generating the model.
        :param directives: A list of directives to apply to the model generation process.
        :return: A GenerationResult object containing the training and inference code, and the predictor module.
        """
        # Check either timeout or max_iterations is set
        if timeout is None and max_iterations is None:
            raise ValueError("Either timeout or max_iterations must be set")

        # Start the model generation run
        run_id = f"run-{datetime.now().isoformat()}".replace(":", "-").replace(".", "-")
        logger.info(f"🔨 Starting model generation with cache {self.filedir}")

        # Define the problem statement to be used; it can change at each call of generate()
        task = join_task_statement(self.intent, self.input_schema, self.output_schema, self.constraints, directives)

        # Select the metric to optimise and the stopping condition for the search
        target_metric = self.plan_generator.select_target_metric(task)
        stop_condition = StoppingCondition(max_iterations, timeout, None)
        logger.info(f"🔨 Optimising {target_metric.name}, {str(stop_condition)}")

        # Initialise the solution graph with initial nodes
        self._initialise_graph(config.model_search.initial_nodes, task, target_metric)
        logger.info(f"🔨 Initialised solution graph with {config.model_search.initial_nodes} nodes")

        # Explore the solution graph until the stopping condition is met
        best_node = self._produce_trained_model(task, run_id, dataset, target_metric, stop_condition)
        self._cache_model_files(best_node)
        logger.info("🧠 Generating inference code for the best solution")
        best_node = self._produce_inference_code(best_node, self.input_schema, self.output_schema)
        self._cache_model_files(best_node)
        logger.info(f"✅ Built predictor for model with performance: {best_node.performance}")

        # compile the inference code into a module
        predictor: types.ModuleType = types.ModuleType("predictor")
        exec(best_node.inference_code, predictor.__dict__)

        return GenerationResult(
            best_node.training_code,
            best_node.inference_code,
            predictor,
            best_node.model_artifacts,
            best_node.performance,
        )

    def _initialise_graph(self, n_nodes: int, task: str, metric: Metric) -> None:
        """
        Creates the initial set of nodes in the solution graph.

        :param n_nodes: The number of nodes to create.
        :return: None
        """
        for _ in range(n_nodes):
            self.graph.add_node(
                node=Node(solution_plan=self.plan_generator.generate_solution_plan(task, metric.name)),
                parent=None,
            )

    def _produce_trained_model(
        self, task: str, run_name: str, dataset: pd.DataFrame, target_metric: Metric, stop_condition: StoppingCondition
    ) -> Node:
        """
        Searches for the best training solution in the solution graph.

        :param task: the problem statement for which to generate a solution
        :param run_name: name of this run, used for working directory
        :param dataset: dataset to be used for training
        :param target_metric: metric to optimise for
        :param stop_condition: determines when the search should stop
        :return: graph node containing the best solution
        """
        start_time = time.time()
        i = 0
        best_metric: Metric = target_metric

        # Explore the solution graph until the stopping condition is met
        while not stop_condition.is_met(i, start_time, best_metric):
            # If we have visited all nodes, expand the graph by adding new nodes
            if not self.graph.unvisited_nodes:
                node_to_expand = self.search_policy.select_node_expand()[0]
                plan = self.plan_generator.generate_solution_plan(task, target_metric.name)
                self.graph.add_node(Node(plan), parent=node_to_expand)

            # Select a node to visit (i.e. evaluate)
            node: Node = self.search_policy.select_node_enter()[0]

            # Generate training code for the selected node
            logger.info(f"🔨 Solution {i} (graph depth {node.depth}): generating training module")
            node.training_code = self.train_generator.generate_training_code(task, node.solution_plan)
            node.visited = True

            # Iteratively validate and fix the training code
            for i_fix in range(config.model_search.max_fixing_attempts_train):
                node.exception_was_raised = False
                node.exception = None

                # Validate the training code, stopping at the first failed validation
                validation = self.train_validators.validate(node.training_code)
                if not validation.passed:
                    logger.warning(f"Node {i}, attempt {i_fix}: Failed validation {validation}")
                    node.exception_was_raised = True
                    node.exception = validation.exception

                    review = self.train_generator.review_training_code(
                        node.training_code, task, node.solution_plan, str(validation)
                    )
                    node.training_code = self.train_generator.fix_training_code(
                        node.training_code, node.solution_plan, review, str(validation)
                    )
                    continue

                # If the code passes all static validations, execute the code
                # TODO: Training can happen in parallel to further exploration
                execute_node(
                    node=node,
                    executor=ProcessExecutor(
                        execution_id=f"{i}-{node.id}-{i_fix}",
                        code=node.training_code,
                        working_dir=f"./workdir/{run_name}/",
                        dataset=dataset,
                        timeout=config.execution.timeout,
                        code_execution_file_name=config.execution.runfile_name,
                    ),
                    metric_to_optimise=target_metric,
                )

                # If the code raised an exception, attempt to fix again
                if node.exception_was_raised:
                    review = self.train_generator.review_training_code(
                        node.training_code, task, node.solution_plan, str(node.exception)
                    )
                    node.training_code = self.train_generator.fix_training_code(
                        node.training_code, node.solution_plan, review, str(node.exception)
                    )
                    continue
                else:
                    break

            # Unpack the solution's performance; if this is better than the best so far, update
            if node.performance and isinstance(node.performance.value, float):
                logger.info(f"🤔 Solution {i} (graph depth {node.depth}) performance: {str(node.performance)}")
                if best_metric is None or node.performance > best_metric:
                    best_metric = node.performance
            else:
                logger.info(
                    f"❌ Solution {i} (graph depth {node.depth}) did not return valid performance: "
                    f"{str(node.performance)}"
                )
            logger.info(
                f"📈 Explored {i + 1}/{stop_condition.max_generations} nodes, best performance so far: {str(best_metric)}"
            )
            i += 1

        valid_nodes = [n for n in self.graph.nodes if n.performance is not None and not n.exception_was_raised]
        if not valid_nodes:
            raise RuntimeError("❌ No valid solutions found during search")
        return max(valid_nodes, key=lambda n: n.performance)

    def _produce_inference_code(self, node: Node, input_schema: dict, output_schema: dict) -> Node:
        """
        Generates inference code for the given node, and validates it.

        :param node: the graph node for which to generate inference code
        :param input_schema: the input schema that the predict function must match
        :param output_schema: the output schema that the predict function must match
        :return: the node with updated inference code
        """
        # Create model directory in .smolcache with proper model ID format
        cache_dir = Path(config.file_storage.model_cache_dir)
        model_id = f"model-{hash(node.id)}-{node.id}"
        model_dir = cache_dir / model_id

        # Create directory if it doesn't exist
        model_dir.mkdir(parents=True, exist_ok=True)

        # Copy model files to the model directory
        for artifact in node.model_artifacts:
            artifact_path = Path(artifact)
            if artifact_path.is_file():
                shutil.copy2(artifact_path, model_dir / artifact_path.name)
            elif artifact_path.is_dir():
                for item in artifact_path.glob("*"):
                    if item.is_file():
                        shutil.copy2(item, model_dir / item.name)

        # Update node's model_artifacts to use the new path
        node.model_artifacts = [str(model_dir)]

        validator = InferenceCodeValidator(
            provider=self.provider,
            intent=self.intent,
            input_schema=input_schema,
            output_schema=output_schema,
            n_samples=10,
            model_id=model_id,
        )

        # Generate inference code using LLM
        node.inference_code = self.infer_generator.generate_inference_code(
            input_schema=input_schema,
            output_schema=output_schema,
            training_code=node.training_code,
            model_id=model_id,
        )

        # Iteratively validate and fix the inference code
        fix_attempts = config.model_search.max_fixing_attempts_predict
        for i in range(fix_attempts):
            node.exception_was_raised = False
            node.exception = None

            # Validate the inference code, stopping at the first failed validation
            validation = validator.validate(node.inference_code)
            if not validation.passed:
                logger.info(f"⚠️ Inference solution {i+1}/{fix_attempts} failed validation, fixing ...")
                node.exception_was_raised = True
                node.exception = validation.exception
                review = self.infer_generator.review_inference_code(
                    inference_code=node.inference_code,
                    input_schema=input_schema,
                    output_schema=output_schema,
                    training_code=node.training_code,
                    problems=str(validation),
                    model_id=model_id,
                )
                node.inference_code = self.infer_generator.fix_inference_code(
                    node.inference_code,
                    review,
                    str(validation),
                    model_id=model_id,
                )
                continue

        if node.exception_was_raised:
            raise RuntimeError(f"❌ Failed to generate valid inference code: {str(node.exception)}")
        return node

    def _cache_model_files(self, node: Node) -> None:
        """
        Copies the model artifacts to the model cache directory, and updates the paths in the code.

        :param node: graph node containing the model artifacts
        :return: None
        """
        # Make sure the model cache directory exists
        self.filedir.mkdir(parents=True, exist_ok=True)

        # Create the model directory with model ID
        model_dir = self.filedir / f"model-{hash(node.id)}-{node.id}"
        model_dir.mkdir(parents=True, exist_ok=True)

        # Copy artifacts to cache and update the code to match
        for i in range(len(node.model_artifacts)):
            path: Path = Path(node.model_artifacts[i])
            name: str = Path(path).name
            try:
                if path.is_dir():
                    # If it's a directory, copy its contents
                    if model_dir.exists():
                        shutil.rmtree(model_dir)
                    shutil.copytree(path, model_dir)
                    # Update code paths
                    if node.training_code:
                        node.training_code = node.training_code.replace(str(path), str(model_dir.as_posix()))
                    if node.inference_code:
                        node.inference_code = node.inference_code.replace(str(path), str(model_dir.as_posix()))
                    node.model_artifacts[i] = model_dir
                else:
                    # If it's a file, copy directly
                    shutil.copy(path, model_dir / name)
                    # Update code paths
                    if node.training_code:
                        node.training_code = node.training_code.replace(name, str((model_dir / name).as_posix()))
                    if node.inference_code:
                        node.inference_code = node.inference_code.replace(name, str((model_dir / name).as_posix()))
                    node.model_artifacts[i] = model_dir / name
            except shutil.SameFileError:
                pass
        # Delete the working directory before returning
        if os.path.exists("./workdir"):
            shutil.rmtree("./workdir")
