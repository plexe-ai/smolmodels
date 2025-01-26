"""
Configuration for the smolmodels library.
"""

from dataclasses import dataclass, field
from typing import List
from string import Template


@dataclass(frozen=True)
class _Config:
    @dataclass(frozen=True)
    class _LoggingConfig:
        level: str = field(default="INFO")
        format: str = field(default="[%(asctime)s - %(name)s - %(levelname)s - (%(threadName)-10s)]: - %(message)s")

    @dataclass(frozen=True)
    class _ModelSearchConfig:
        initial_nodes: int = field(default=1)
        max_nodes: int = field(default=5)
        max_fixing_attempts: int = field(default=2)

    @dataclass(frozen=True)
    class _ExecutionConfig:
        timeout: int = field(default=3600)
        runfile_name: str = field(default="execution_script.py")

    @dataclass(frozen=True)
    class _CodeGenerationConfig:
        allowed_packages: List[str] = field(default_factory=lambda: ["pandas", "numpy", "sklearn"])
        k_fold_validation: int = field(default=5)
        # prompts used in generating plans or making decisions
        prompt_planning_base: Template = field(
            default=Template("Experienced ML Engineer competing in a Kaggle competition.")
        )
        prompt_planning_select_metric: Template = field(default=Template("Select the metric to optimise for the task."))
        prompt_planning_select_stop_condition: Template = field(
            default=Template("Define the stopping condition for the task.")
        )
        prompt_planning_generate_plan: Template = field(
            default=Template(
                "Experienced ML Engineer competing in a Kaggle competition. Design a solution plan.\n\n"
                "**Task:** ${problem_statement}\n\n"
                "**Memory:** ${context}\n\n"
                "Outline the solution (3-5 sentences) and include a markdown code block that implements the solution "
                "and prints the evaluation metric. Use ${allowed_packages}. Avoid ensembling and hyperparameter tuning."
            )
        )
        # prompts used in generating, fixing or reviewing training code
        prompt_training_base: Template = field(
            default=Template("You are an experienced ML Engineer developing a Kaggle solution.")
        )
        prompt_training_generate: Template = field(
            default=Template(
                "Write a Python script to solve the task based on the plan.\n\n"
                "# Task: ${problem_statement}\n"
                "# Plan: ${plan}\n"
                "# History: ${history}\n\n"
                "Do not write any explanation of your approach. Only return the code. "
                "Train the model, compute and print the evaluation metric, and save the model as 'model.joblib' in './working'. "
                "Use ${allowed_packages}. Do not skip steps or combine preprocessors and models in the same joblib file."
            )
        )
        prompt_training_fix: Template = field(
            default=Template(
                "Fix the previous solution based on the following information.\n\n"
                "# Plan: ${plan}\n"
                "# Code: ${training_code}\n"
                "# Issues: ${review}\n"
                "# Errors: ${problems}\n"
                "# History: ${history}\n\n"
                "Correct the code, train the model, compute and print the evaluation metric, and save the model in './working'."
            )
        )
        prompt_training_review: Template = field(
            default=Template(
                "Review the solution to enhance test performance and fix issues.\n\n"
                "# Task: ${problem_statement}\n"
                "# Plan: ${plan}\n"
                "# Code: ${training_code}\n"
                "# Errors: ${problems}\n"
                "# History: ${history}\n\n"
                "Suggest a single, actionable improvement considering previous reviews."
            )
        )
        # prompts used in generating, fixing or reviewing prediction code
        prompt_inference_base: Template = field(default=Template("Experienced ML Engineer deploying a trained model."))
        prompt_inference_generate: Template = field(
            default=Template(
                "Write an inference script.\n\n"
                "# Task: ${problem_statement}\n"
                "# Plan: ${plan}\n"
                "# Code: ${training_code}\n"
                "# Context: ${context}\n\n"
                "Load 'model.joblib' from './working', preprocess inputs, and generate predictions. Save results as 'predictions.csv'. "
                "Handle errors gracefully. Use ${allowed_packages}."
            )
        )
        prompt_inference_fix: Template = field(
            default=Template(
                "Fix the inference script to resolve issues.\n\n"
                "# Code: ${inference_code}\n"
                "# Issues: ${review}\n"
                "# Errors: ${problems}\n\n"
                "Ensure compatibility with the training pipeline and handle errors gracefully."
            )
        )
        prompt_inference_review: Template = field(
            default=Template(
                "Review the inference script for improvements.\n\n"
                "# Task: ${problem_statement}\n"
                "# Plan: ${plan}\n"
                "# Code: ${inference_code}\n"
                "# Context: ${context}\n\n"
                "Provide optimisation and error-handling suggestions."
            )
        )

    @dataclass(frozen=True)
    class _DataGenerationConfig:
        pass  # todo: implement

    # configuration objects
    logging: _LoggingConfig = field(default_factory=_LoggingConfig)
    model_search: _ModelSearchConfig = field(default_factory=_ModelSearchConfig)
    code_generation: _CodeGenerationConfig = field(default_factory=_CodeGenerationConfig)
    execution: _ExecutionConfig = field(default_factory=_ExecutionConfig)
    data_generation: _DataGenerationConfig = field(default_factory=_DataGenerationConfig)


# Instantiate configuration
def load_config() -> _Config:
    return _Config()


config: _Config = load_config()
