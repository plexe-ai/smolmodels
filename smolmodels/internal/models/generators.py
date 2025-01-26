"""
This module provides the main function `generate` for generating machine learning models based on
a given problem statement, input schema, and output schema. The function explores the solution space,
generates training and inference code, and returns callable functions for training and prediction.

Functions:
    generate: Generates training and inference code for a given problem statement and schemas.

Constants:
    NUMBER_INITIAL_NODES: The number of initial nodes to add to the solution graph.
    MAX_FIXING_ATTEMPTS: The maximum number of attempts to fix generated code.

"""

import time
from typing import List, Optional, Tuple, Callable

from smolmodels.config import config
from smolmodels.callbacks import Callback
from smolmodels.constraints import Constraint
from smolmodels.directives import Directive
from smolmodels.internal.models.entities.graph import Graph
from smolmodels.internal.models.entities.metric import Metric
from smolmodels.internal.models.entities.node import Node
from smolmodels.internal.models.entities.stopping_condition import StoppingCondition
from smolmodels.internal.models.execution.executor import Executor
from smolmodels.internal.models.execution.process_executor import ProcessExecutor
from smolmodels.internal.models.generation.inference import (
    generate_inference_code,
    generate_inference_tests,
    fix_inference_code,
    review_inference_code,
)
from smolmodels.internal.models.generation.planning import (
    generate_solution_plan,
    select_metric_to_optimise,
    select_stopping_condition,
)
from smolmodels.internal.models.generation.training import (
    generate_training_code,
    generate_training_tests,
    fix_training_code,
    review_training_code,
)
from smolmodels.internal.models.search.policy import SearchPolicy
from smolmodels.internal.models.search.random_policy import RandomSearchPolicy
from smolmodels.internal.models.validation.security import SecurityValidator
from smolmodels.internal.models.validation.syntax import SyntaxValidator
from smolmodels.internal.models.validation.validator import Validator
import smolmodels.internal.models.utils as sm_utils


def generate(
    intent: str,
    input_schema: dict,
    output_schema: dict,
    dataset: str,
    constraints: List[Constraint] = None,
    directives: List[Directive] = None,
    callbacks: List[Callback] = None,
    isolation: str = "process",
    executor: Optional[Executor] = None,
    search_policy: Optional[SearchPolicy] = None,
) -> Tuple[Callable, Callable]:
    """
    Generate training and inference code for a given problem statement and schemas.

    Args:
        intent (str): The description or intent of the problem.
        input_schema (dict): A dictionary defining the schema of the input data.
        output_schema (dict): A dictionary defining the schema of the output data.
        dataset (str): The dataset to be used for training.
        constraints (List[Constraint], optional): Constraints to be applied to the model generation process. Defaults to None.
        directives (List[Directive], optional): Directives to guide the model generation process. Defaults to None.
        callbacks (List[Callback], optional): Callbacks to execute during model generation. Defaults to None.
        isolation (str, optional): The isolation method for execution (e.g., "process"). Defaults to "process".
        executor (Optional[Executor], optional): Executor for running generated code. Defaults to None.
        search_policy (Optional[SearchPolicy], optional): Policy to guide exploration of the solution graph. Defaults to None.

    Returns:
        Tuple[Callable, Callable]: A tuple containing the training function and the prediction function.
    """
    # Join the problem statement into a single string
    problem_statement: str = sm_utils.join_problem_statement(
        intent, input_schema, output_schema, constraints, directives
    )

    # Decide what metric to optimise based on the definition of the problem
    metric_to_optimise: Metric = select_metric_to_optimise(problem_statement, dataset)
    stopping_condition: StoppingCondition = select_stopping_condition(problem_statement, dataset)

    # Create the solution graph with initial nodes
    graph: Graph = Graph()
    search_policy: SearchPolicy = search_policy or RandomSearchPolicy(graph)

    # Create classes used in code generation and review
    validators: List[Validator] = [SyntaxValidator(), SecurityValidator()]

    for i in range(config.model_search.initial_nodes):
        graph.add_node(Node(solution_plan=generate_solution_plan(problem_statement)), None)

    # Explore the solution space until the stopping condition is met
    i: int = 0
    best_metric: Metric = metric_to_optimise

    while not stopping_condition.is_met(i, time.time(), best_metric):
        # Expand the graph by selecting a node to explore out from
        if i != 0:
            node_expand: Node = search_policy.select_node_expand()[0]
            graph.add_node(
                Node(
                    solution_plan=generate_solution_plan(
                        problem_statement=problem_statement,
                        context=str(node_expand),  # TODO: Pass a proper summary of the parent node (or the whole graph)
                    )
                ),
            )

        # Select a node to evaluate using the search policy
        node: Node = search_policy.select_node_enter()[0]

        # Generate the code for the node
        node.training_code = generate_training_code(problem_statement, node.solution_plan)
        node.training_tests = generate_training_tests(problem_statement, node.solution_plan, node.training_code)

        # Review the generated training code
        for _ in range(config.model_search.max_fixing_attempts):
            for validator in validators:
                result = validator.validate(node.training_code)
                if not result.passed:
                    review = review_training_code(
                        node.training_code, problem_statement, node.solution_plan, str(result)
                    )
                    node.training_code = fix_training_code(node.training_code, review)
                    continue

        # TODO: Training can happen in parallel to further exploration
        sm_utils.execute_node(node, ProcessExecutor(node.training_code, "./workdir"))

        # If this node achieved a better metric, update the best metric
        i += 1
        best_metric = max(best_metric, node.metric)

    # Generate the inference code for the best node
    best_node: Node = graph.nodes.sort(key=lambda n: n.metric)[-1]
    best_node.inference_code = generate_inference_code(
        problem_statement, best_node.solution_plan, best_node.training_code
    )
    best_node.inference_tests = generate_inference_tests(
        problem_statement, best_node.solution_plan, best_node.training_code, best_node.training_code
    )

    # Review the generated inference code
    for _ in range(config.model_search.max_fixing_attempts):
        for validator in validators:
            result = validator.validate(best_node.inference_code)
            if not result.passed:
                review = review_inference_code(
                    best_node.inference_code, problem_statement, best_node.solution_plan, str(result)
                )
                best_node.inference_code = fix_inference_code(best_node.inference_code, review, str(result))
                continue

    # Write out the training and inference code and return the compiled functions
    # TODO: Check this actually works
    trainer: Callable = eval(compile(best_node.training_code, best_node.model_artifacts.training_code, "eval"))
    predictor: Callable = eval(compile(best_node.inference_code, best_node.model_artifacts.inference_code, "eval"))

    return trainer, predictor
