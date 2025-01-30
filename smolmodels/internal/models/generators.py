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

import logging
import shutil
import time
import types
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

import smolmodels.internal.models.utils as sm_utils
from smolmodels.callbacks import Callback
from smolmodels.config import config
from smolmodels.constraints import Constraint
from smolmodels.directives import Directive
from smolmodels.internal.common.providers.provider import Provider
from smolmodels.internal.models.entities.graph import Graph
from smolmodels.internal.models.entities.metric import Metric
from smolmodels.internal.models.entities.node import Node
from smolmodels.internal.models.entities.stopping_condition import StoppingCondition
from smolmodels.internal.models.execution.executor import Executor
from smolmodels.internal.models.execution.process_executor import ProcessExecutor
from smolmodels.internal.models.generation.inference import InferenceCodeGenerator
from smolmodels.internal.models.generation.planning import SolutionPlanGenerator
from smolmodels.internal.models.generation.training import TrainingCodeGenerator
from smolmodels.internal.models.search.policy import SearchPolicy
from smolmodels.internal.models.search.random_policy import RandomSearchPolicy
from smolmodels.internal.models.validation.security import SecurityValidator
from smolmodels.internal.models.validation.syntax import SyntaxValidator
from smolmodels.internal.models.validation.validator import Validator, ValidationResult

logger = logging.getLogger(__name__)

# todo: where to move these?
logging.getLogger("httpcore.http11").setLevel(logging.WARNING)
logging.getLogger("httpcore.connection").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai._base_client").setLevel(logging.WARNING)


def generate(
    intent: str,
    input_schema: dict,
    output_schema: dict,
    dataset: pd.DataFrame,
    provider: Provider,
    filedir: Path,
    constraints: List[Constraint] = None,
    directives: List[Directive] = None,
    callbacks: List[Callback] = None,
    isolation: str = "process",
    executor: Optional[Executor] = None,
    search_policy: Optional[SearchPolicy] = None,
    timeout: int = config.model_search.max_time_elapsed,
) -> Tuple[types.ModuleType, str, types.ModuleType, str, List[Path | str], str]:
    """
    Generate training and inference code for a given problem statement and schemas.

    :param intent: The description or intent of the problem.
    :param input_schema: A dictionary defining the schema of the input data.
    :param output_schema: A dictionary defining the schema of the output data.
    :param dataset: The dataset to be used for training.
    :param provider: The provider to use for model generation.
    :param filedir: The directory to save the model artifacts.
    :param constraints: Constraints to be applied to the model generation process. Defaults to None.
    :param directives: Directives to guide the model generation process. Defaults to None.
    :param callbacks: Callbacks to execute during model generation. Defaults to None.
    :param isolation: The isolation method for execution (e.g., "process"). Defaults to "process".
    :param executor: Executor for running generated code. Defaults to None.
    :param search_policy: Policy to guide exploration of the solution graph. Defaults to None.
    :param timeout: The maximum time allowed for model generation. Defaults to config.model_search.max_time_elapsed.
    :return: A tuple containing the training function and the prediction function.
    """
    # Set up the model generation process
    start_time: float = time.time()
    run_name = f"run-{datetime.now().isoformat()}".replace(":", "-").replace(".", "-")
    print(f"🔨 Starting model generation with cache {filedir}")

    # Create code generators used to produce plans, training code, and inference code
    plan_generator = SolutionPlanGenerator(provider)
    train_generator = TrainingCodeGenerator(provider)
    infer_generator = InferenceCodeGenerator(provider)

    # Join the problem statement into a single string
    problem_statement: str = sm_utils.join_problem_statement(
        intent, input_schema, output_schema, constraints, directives
    )

    # Decide what metric to optimise based on the definition of the problem
    metric_to_optimise: Metric = plan_generator.select_metric_to_optimise(problem_statement)
    stopping_condition: StoppingCondition = plan_generator.select_stopping_condition(
        problem_statement=problem_statement,
        metric=metric_to_optimise,
        max_iterations=config.model_search.max_nodes,
        max_time=timeout,
    )
    print(f"🔨 Optimising {metric_to_optimise.name}; {str(stopping_condition)}")

    # Create the solution graph with initial nodes
    graph: Graph = Graph()
    search_policy: SearchPolicy = search_policy or RandomSearchPolicy(graph)

    # Create classes used in code generation and review
    validators: List[Validator] = [SyntaxValidator(), SecurityValidator()]

    for _ in tqdm(range(config.model_search.initial_nodes), desc="🔨 Initialising solution graph", colour="red"):
        graph.add_node(
            Node(
                solution_plan=plan_generator.generate_solution_plan(
                    problem_statement=problem_statement, metric_to_optimise=metric_to_optimise.name
                )
            ),
            parent=None,
        )

    # Explore the solution space until the stopping condition is met
    i: int = 0
    best_metric: Metric = metric_to_optimise

    while not stopping_condition.is_met(i, start_time, best_metric):
        # Expand the graph by selecting a node to explore out from
        if i != 0:
            node_expand: Node = search_policy.select_node_expand()[0]
            graph.add_node(
                Node(
                    solution_plan=plan_generator.generate_solution_plan(
                        problem_statement=problem_statement,
                        metric_to_optimise=metric_to_optimise.name,
                    )
                ),
                parent=node_expand,
            )

        # Select a node to evaluate using the search policy
        node: Node = search_policy.select_node_enter()[0]

        # Generate the code for the node
        node.training_code = train_generator.generate_training_code(problem_statement, node.solution_plan)
        node.visited = True
        # node.training_tests = generate_training_tests(problem_statement, node.solution_plan, node.training_code)

        # Review the generated training code
        for i_fix in tqdm(
            range(config.model_search.max_fixing_attempts_train),
            desc=f"🔨 Node {i} (depth {node.depth}) | Reviewing and training",
            colour="red",
        ):
            result: ValidationResult | None = None

            for validator in validators:
                result = validator.validate(node.training_code)
                if not result.passed:
                    logger.warning(f"Node {i}, attempt {i_fix}: Failed validation {result}")
                    break

            if not result.passed:
                review = train_generator.review_training_code(
                    node.training_code, problem_statement, node.solution_plan, str(result)
                )
                node.training_code = train_generator.fix_training_code(
                    node.training_code, node.solution_plan, review, str(result)
                )
                continue

            # If the code passes all static validations, execute the code
            # TODO: Training can happen in parallel to further exploration
            sm_utils.execute_node(
                node=node,
                executor=ProcessExecutor(
                    execution_id=f"{i}-{node.id}-{i_fix}",
                    code=node.training_code,
                    working_dir=f"./workdir/{run_name}/",
                    dataset=dataset,
                    timeout=config.execution.timeout,
                    code_execution_file_name=config.execution.runfile_name,
                ),
                metric_to_optimise=metric_to_optimise,
            )

            # If the code raised an exception, attempt to fix again
            if node.exception_was_raised:
                review = train_generator.review_training_code(
                    node.training_code, problem_statement, node.solution_plan, str(node.exception)
                )
                node.training_code = train_generator.fix_training_code(
                    node.training_code, node.solution_plan, review, str(node.exception)
                )
                continue
            else:
                break

        i += 1
        # Unpack the solution's performance; if this is better than the best so far, update
        if node.performance and isinstance(node.performance.value, float):
            print(f"🤔 Node {i} (depth {node.depth}) performance: {str(node.performance)}")
            if best_metric is None or node.performance > best_metric:
                best_metric = node.performance
        else:
            print(
                f"❌ Node {i} (depth {node.depth}) did not return a valid performance metric: {str(node.performance)}"
            )
        print(
            f"📈 Explored {i}/{stopping_condition.max_generations} nodes, best performance so far: {str(best_metric)}"
        )

    valid_nodes = [n for n in graph.nodes if n.performance is not None and not n.exception_was_raised]
    if not valid_nodes:
        raise RuntimeError("No valid solutions found during search")

    # Generate the inference code for the best node
    print("🧠 Generating inference code for the best solution")
    best_node: Node = max(valid_nodes, key=lambda n: n.performance)
    best_node.inference_code = infer_generator.generate_inference_code(
        input_schema=input_schema,
        output_schema=output_schema,
        training_code=best_node.training_code,
    )
    # best_node.inference_tests = generate_inference_tests(
    #     problem_statement, best_node.solution_plan, best_node.training_code, best_node.training_code
    # )

    # Review the generated inference code
    for _ in tqdm(
        range(config.model_search.max_fixing_attempts_train), desc="🔨 Reviewing and predicting", colour="red"
    ):
        result: ValidationResult | None = None

        for validator in validators:
            result = validator.validate(best_node.inference_code)
            if not result.passed:
                logger.warning(f"Attempt {_} | code failed validation: {result}")
                break

        if not result.passed:
            review = infer_generator.review_inference_code(
                inference_code=best_node.inference_code,
                input_schema=input_schema,
                output_schema=output_schema,
                training_code=best_node.training_code,
                problems=str(result),
            )
            best_node.inference_code = infer_generator.fix_inference_code(best_node.inference_code, review, str(result))
            continue

    print(f"✅ Built predictor for model with performance: {best_node.performance}")
    # Copy model artifacts to the model cache directory, and update paths in code
    filedir.mkdir(parents=True, exist_ok=True)
    for i in range(len(best_node.model_artifacts)):
        artifact: Path = Path(best_node.model_artifacts[i])
        basename: str = Path(artifact).name
        shutil.copy(artifact, filedir)
        best_node.training_code = best_node.training_code.replace(basename, str((filedir / basename).as_posix()))
        best_node.inference_code = best_node.inference_code.replace(basename, str((filedir / basename).as_posix()))
        best_node.model_artifacts[i] = (filedir / basename).as_posix()

    # Write out the training and inference code and return the compiled functions
    trainer: types.ModuleType = types.ModuleType("smoltrainer")
    predictor: types.ModuleType = types.ModuleType("smolpredictor")
    exec(best_node.training_code, trainer.__dict__)
    exec(best_node.inference_code, predictor.__dict__)

    # Delete the working directory before returning
    shutil.rmtree("./workdir")

    return (
        trainer,
        best_node.training_code,
        predictor,
        best_node.inference_code,
        best_node.model_artifacts,
        str(best_node.performance),
    )
