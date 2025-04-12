"""
This module provides a model generation system powered by an agent-based architecture.

The main class `ModelBuilderAgentOrchestrator` facilitates the generation of machine learning models
based on a given problem statement, input schema, and output schema. Using an agentic approach
with the smolagents library, it orchestrates the exploration of the solution space,
generates training and inference code, and returns callable functions for training and prediction.

The agent-based architecture allows for a more flexible and iterative approach to model building,
with specialized tools for different aspects of the generation process such as solution planning,
code generation, validation, and performance evaluation.
"""

import logging
import os
import types
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Type, Dict

from pydantic import BaseModel
from smolagents import CodeAgent, LiteLLMModel

from smolmodels.config import prompt_templates, config
from smolmodels.constraints import Constraint
from smolmodels.internal.common.datasets.interface import TabularConvertible
from smolmodels.internal.common.provider import Provider
from smolmodels.internal.common.registries.datasets import DatasetRegistry
from smolmodels.internal.common.registries.artifacts import ArtifactRegistry
from smolmodels.internal.common.registries.callbacks import CallbackRegistry
from smolmodels.internal.models.entities.metric import Metric, MetricComparator, ComparisonMethod
from smolmodels.internal.models.interfaces.predictor import Predictor
from smolmodels.internal.models.tools.execution_tools import execute_training_code
from smolmodels.internal.models.tools.generation_tools import (
    generate_solution_plan,
    generate_training_code,
    generate_inference_code,
    fix_training_code,
    fix_inference_code,
    review_training_code,
    review_inference_code,
)
from smolmodels.internal.models.tools.metrics_tools import select_target_metric
from smolmodels.internal.models.tools.evaluation_tools import review_model
from smolmodels.internal.models.tools.validation_tools import (
    validate_training_code,
    validate_inference_code,
)
from smolmodels.internal.common.utils.prompt_utils import join_task_statement

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="smolagents")


@dataclass
class GenerationResult:
    training_source_code: str
    inference_source_code: str
    predictor: Predictor
    model_artifacts: List[Path]
    performance: Metric  # Validation performance
    test_performance: Metric = None  # Test set performance
    metadata: Dict[str, str] = field(default_factory=dict)  # Model metadata


class ModelBuilderAgentOrchestrator:
    """
    Encapsulates the process of generating machine learning models based on a given problem statement.
    The model generator sets up the code generators and other dependencies required to
    explore solution options. It generates training and inference code, and returns a callable predictor.

    This implementation uses an agent-based approach powered by the smolagents library.

    Attributes:
        intent: The intent of the model to generate.
        input_schema: The input schema for the model.
        output_schema: The output schema for the model.
        provider: The provider to use for generating models.
        constraints: A list of constraints to apply to the model.

    Example:
    >>> from smolmodels.internal.models.agents import ModelBuilderAgentOrchestrator
    >>> ...
    >>> generator = ModelBuilderAgentOrchestrator(
    >>>     intent="classify",
    >>>     input_schema=create_model("input", {"age": int}),
    >>>     output_schema=create_model("output", {"label": str}),
    >>>     provider=Provider(),
    >>> )
    """

    def __init__(
        self,
        intent: str,
        input_schema: Type[BaseModel],
        output_schema: Type[BaseModel],
        provider: Provider,
        constraints: List[Constraint] = None,
        verbose: bool = False,
    ) -> None:
        """
        Initialises the model generator with the given problem statement, input schema, and output schema.

        :param intent: The intent of the model to generate.
        :param input_schema: The input schema for the model.
        :param output_schema: The output schema for the model.
        :param provider: The provider to use for generating models.
        :param constraints: A list of constraints to apply to the model.
        :param verbose: Whether to display detailed agent logs (default: False).
        """
        # Set up the basic configuration of the model generator
        self.intent: str = intent
        self.input_schema: Type[BaseModel] = input_schema
        self.output_schema: Type[BaseModel] = output_schema
        self.constraints: List[Constraint] = constraints or []
        self.provider: Provider = provider
        self.isolation: str = "subprocess"
        self.run_timeout = None
        self.verbose: bool = verbose

        # Store current model artifacts registry names for validation
        self.current_artifact_names: List[str] = []

        # Initialize agent configuration based on provider
        self.agent_model = LiteLLMModel(model_id=self.provider.model)
        self.agent = self._create_model_builder_agent()

    def generate(
        self,
        datasets: Dict[str, TabularConvertible],
        run_timeout: int,
        timeout: int = None,
        max_iterations=None,
        callbacks=None,
    ) -> GenerationResult:
        """
        Generates a machine learning model based on the given problem statement, input schema, and output schema.
        Uses an agent-based approach to explore the solution space and generate the model.

        :param datasets: The dataset to use for training the model.
        :param timeout: The maximum total time to spend generating the model, in seconds (all iterations combined).
        :param max_iterations: The maximum number of iterations to spend generating the model.
        :param run_timeout: The maximum time to spend on each individual model training run, in seconds.
        :param callbacks: list of callbacks to notify during the model building process.
        :return: A GenerationResult object containing the training and inference code, and the predictor module.
        """
        # Store the individual_run_timeout for later use if provided
        self.run_timeout = run_timeout

        # Start the model generation run
        run_id = f"run-{datetime.now().isoformat()}".replace(":", "-").replace(".", "-")
        working_dir = f"./workdir/{run_id}/"

        # Create working directory if it doesn't exist
        os.makedirs(working_dir, exist_ok=True)

        # Initialize dataset registry
        dataset_registry = DatasetRegistry()

        # Reset artifact tracking
        # We don't need to initialize ArtifactRegistry here as it's a singleton
        # It will be initialized when needed by the register_model_artifacts callback
        self.current_artifact_names = []

        # Register original datasets
        for name, dataset in datasets.items():
            dataset_registry.register(name, dataset)

        # Split datasets into train, validation, and test sets
        train_dataset_names = []
        validation_dataset_names = []
        test_dataset_names = []

        logger.info("🔪 Splitting datasets into train, validation, and test sets")
        for name, dataset in datasets.items():
            train_ds, val_ds, test_ds = dataset.split(train_ratio=0.9, val_ratio=0.1, test_ratio=0.0)

            # Register split datasets in the registry
            train_name = f"{name}_train"
            val_name = f"{name}_val"
            test_name = f"{name}_test"

            dataset_registry.register(train_name, train_ds)
            dataset_registry.register(val_name, val_ds)
            dataset_registry.register(test_name, test_ds)

            # Store dataset names for agent
            train_dataset_names.append(train_name)
            validation_dataset_names.append(val_name)
            test_dataset_names.append(test_name)
            logger.info(
                f"✅  Split dataset {name} into train/validation/test with sizes {len(train_ds)}/{len(val_ds)}/{len(test_ds)}"
            )

        # Define the problem statement to be used
        task = join_task_statement(self.intent, self.input_schema, self.output_schema)

        # Define the agent prompt
        agent_prompt = prompt_templates.agent_builder_prompt(
            task=task,
            train_datasets=train_dataset_names,
            validation_datasets=validation_dataset_names,
            max_iterations=max_iterations,
            timeout=timeout,
            run_timeout=run_timeout,
        )

        # Initialize and register callbacks in the CallbackRegistry
        callback_registry = CallbackRegistry()
        callback_registry.clear()  # Clear any previous callbacks
        callback_registry.register_batch(callbacks)

        # Run callbacks for build start
        for callback in callback_registry.get_all():
            try:
                from smolmodels.callbacks import BuildStateInfo

                # Note: callbacks still receive the actual dataset objects for backward compatibility
                callback.on_build_start(
                    BuildStateInfo(
                        intent=self.intent,
                        input_schema=self.input_schema,
                        output_schema=self.output_schema,
                        provider=self.provider.model,
                        run_timeout=self.run_timeout,
                        max_iterations=max_iterations,
                        timeout=timeout,
                        datasets={name: dataset_registry.get(name) for name in train_dataset_names},
                    )
                )
            except Exception as e:
                logger.warning(f"Error in callback {callback.__class__.__name__}.on_build_start: {e}")

        try:
            # Convert input and output schemas to dictionaries for the agent tools
            from smolmodels.internal.common.utils.pydantic_utils import convert_schema_to_type_dict
            import pandas as pd

            # Create schema type dictionaries for tools
            input_schema_dict = convert_schema_to_type_dict(self.input_schema)
            output_schema_dict = convert_schema_to_type_dict(self.output_schema)

            # Extract input sample from the datasets for inference code validation
            try:
                # Concatenate all train datasets and extract relevant columns for the input schema
                input_sample_dfs = []
                for dataset_name in train_dataset_names:
                    dataset = dataset_registry.get(dataset_name)
                    df = dataset.to_pandas().head(5)  # Get a few rows from each dataset
                    input_sample_dfs.append(df)

                if input_sample_dfs:
                    # Combine datasets and filter for input schema columns
                    combined_df = pd.concat(input_sample_dfs, axis=0).reset_index(drop=True)
                    # Keep only columns that match the input schema
                    input_sample_df = combined_df[list(input_schema_dict.keys())].head(10)

                    # Register the input sample in the registry for validation tool to use
                    dataset_registry.register("predictor_input_sample", input_sample_df)
                    logger.info(f"✅ Registered input sample with {len(input_sample_df)} rows for inference validation")
                else:
                    logger.warning("⚠️ No datasets available to create input sample for validation")
            except Exception as e:
                logger.warning(f"⚠️ Error creating input sample for validation: {str(e)}")

            # Artifact registration is now done directly in the execute_training_code tool

            # Run the agent
            logger.info("🤖 Starting model builder agent")
            result = self.agent.run(
                agent_prompt,
                additional_args={
                    "task": task,
                    "train_datasets": train_dataset_names,
                    "validation_datasets": validation_dataset_names,
                    "working_dir": working_dir,
                    "run_id": run_id,
                    "provider": self.provider.model,
                    "input_schema": input_schema_dict,
                    "output_schema": output_schema_dict,
                    "model_artifact_names": self.current_artifact_names,
                    # Add information for callback context
                    "intent": self.intent,
                    "max_iterations": max_iterations,
                    "timeout": timeout,
                    "run_timeout": run_timeout,
                },
            )

            # Process agent result
            generation_result = self._process_agent_result(result)

            # Run callbacks for build end
            for callback in callback_registry.get_all():
                try:
                    from smolmodels.callbacks import BuildStateInfo

                    # Note: callbacks still receive the actual dataset objects for backward compatibility
                    callback.on_build_end(
                        BuildStateInfo(
                            intent=self.intent,
                            input_schema=self.input_schema,
                            output_schema=self.output_schema,
                            provider=self.provider.model,
                            run_timeout=self.run_timeout,
                            max_iterations=max_iterations,
                            timeout=timeout,
                            datasets={name: dataset_registry.get(name) for name in train_dataset_names},
                        )
                    )
                except Exception as e:
                    logger.warning(f"Error in callback {callback.__class__.__name__}.on_build_end: {e}")

            return generation_result

        except Exception as e:
            logger.error(f"Error during model generation: {str(e)}")
            raise RuntimeError(f"❌ Failed to generate model: {str(e)}") from e

    def _create_model_builder_agent(self) -> CodeAgent:
        """Create the model builder agent with appropriate tools."""
        # Define the tools the agent can use
        tools = [
            generate_solution_plan,
            select_target_metric,
            generate_training_code,
            review_training_code,
            fix_training_code,
            validate_training_code,
            execute_training_code,
            generate_inference_code,
            review_inference_code,
            fix_inference_code,
            validate_inference_code,
            review_model,
        ]

        # Create the agent
        return CodeAgent(
            model=self.agent_model,
            tools=tools,
            add_base_tools=False,
            max_steps=30,  # Allow sufficient steps for exploration
            additional_authorized_imports=config.code_generation.authorized_agent_imports,
            verbosity_level=2 if self.verbose else 0,
        )

    @staticmethod
    def _process_agent_result(result: dict) -> GenerationResult:
        """
        Process the result from the agent run to create a GenerationResult object.
        """
        # In smolagents, the run method returns a structured dict with the final output

        try:
            # Only log the full result when in verbose mode
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Agent result: %s", result)

            # Extract data from the agent result
            training_code = result.get("training_code", "")
            inference_code = result.get("inference_code", "")

            # Extract performance metrics
            if "performance" in result and isinstance(result["performance"], dict):
                metrics = result["performance"]
            else:
                metrics = {}

            metric_name = metrics.get("name", "unknown")
            metric_value = metrics.get("value", 0.0)
            comparison_str = metrics.get("comparison_method", "")
            comparison_method_map = {
                "HIGHER_IS_BETTER": ComparisonMethod.HIGHER_IS_BETTER,
                "LOWER_IS_BETTER": ComparisonMethod.LOWER_IS_BETTER,
                "TARGET_IS_BETTER": ComparisonMethod.TARGET_IS_BETTER,
            }
            comparison_method = ComparisonMethod.HIGHER_IS_BETTER  # Default to higher is better
            for key, method in comparison_method_map.items():
                if key in comparison_str:
                    comparison_method = method

            comparator = MetricComparator(comparison_method)
            performance = Metric(
                name=metric_name,
                value=metric_value,
                comparator=comparator,
            )

            # Get model artifacts from registry or result
            artifact_registry = ArtifactRegistry()
            artifact_names = result.get("model_artifact_names", [])

            artifacts = []
            if artifact_names:
                # Get paths from registry using names
                artifacts = [a for a in artifact_registry.get_multiple(artifact_names).values()]

            # Model metadata
            metadata = result.get("metadata", {"model_type": "unknown", "framework": "unknown"})

            # Compile the inference code into a module
            inference_module: types.ModuleType = types.ModuleType("predictor")
            exec(inference_code, inference_module.__dict__)
            # Instantiate the predictor class from the loaded module
            predictor_class = getattr(inference_module, "PredictorImplementation")
            predictor = predictor_class(artifacts)

            # Convert artifacts to Path objects
            artifact_paths = [Path(artifact) if isinstance(artifact, str) else artifact for artifact in artifacts]

            return GenerationResult(
                training_source_code=training_code,
                inference_source_code=inference_code,
                predictor=predictor,
                model_artifacts=artifact_paths,
                performance=performance,
                test_performance=performance,  # Using the same performance for now
                metadata=metadata,
            )
        except Exception as e:
            raise RuntimeError(f"❌ Failed to process agent result: {str(e)}") from e


# TODO: Future enhancements:
# 1. Add option for direct code generation vs tool-based approach
# 2. Implement specialized sub-agents for different tasks
# 3. Add more sophisticated agent coordination
# 4. Improve agent memory management
# 5. Add agent logging and visualization
