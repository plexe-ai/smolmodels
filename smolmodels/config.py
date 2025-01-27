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
            default=Template("You are an experienced ML Engineer planning a solution to a Kaggle competition.")
        )
        prompt_planning_select_metric: Template = field(default=Template("Select the metric to optimise for the TASK."))
        prompt_planning_select_stop_condition: Template = field(
            default=Template(
                "Define the stopping condition for when we should stop searching for new solutions, "
                "given the following task description, and the metric we are trying to optimize. In deciding, "
                "consider the complexity of the problem, how many solutions it might be reasonable to try, and "
                "what the metric value should be to consider a solution good enough.\n\n"
                "The metric to optimise is ${metric}.\n\n"
                "The task is:\n${problem_statement}\n\n"
            )
        )
        prompt_planning_generate_plan: Template = field(
            default=Template(
                "Write a solution plan for the machine learning problem outlined below.\n\n"
                "# TASK:\n${problem_statement}\n\n"
                "# PREVIOUS ATTEMPTS, IF ANY:**\n${context}\n\n"
                "The solution concept should be explained in 3-5 sentences. Do not include an implementation of the "
                "solution, though you can include small code snippets if relevant to explain the plan. "
                "Do not suggest doing EDA, ensembling, or hyperparameter tuning. "
                "The solution should be feasible using only ${allowed_packages}, and no other non-standard libraries."
            )
        )
        # prompts used in generating, fixing or reviewing training code
        prompt_training_base: Template = field(
            default=Template(
                "You are an experienced ML Engineer implementing a training script for a Kaggle competition."
            )
        )
        prompt_training_generate: Template = field(
            default=Template(
                "Write a Python script to train a machine learning model that solves the TASK outlined below, "
                "using the approach outlined in the plan below.\n\n"
                "# TASK:\n${problem_statement}\n\n"
                "# PLAN:\n${plan}\n"
                "# PREVIOUS ATTEMPTS, IF ANY:\n${history}\n\n"
                "Only return the code to train the model, no explanations outside the code. Any explanation should "
                "be in the comments in the code itself, but your overall answer must only consist of the code script. "
                "The script must train the model, compute and print the final evaluation metric to standard output, "
                "and save the model as 'model.joblib' in the './working' directory. Use only ${allowed_packages}."
                "Train the model, compute and print the evaluation metric, and save the model as 'model.joblib' in './working'. "
                "Use ${allowed_packages}. Do not skip steps or combine preprocessors and models in the same joblib file."
            )
        )
        prompt_training_fix: Template = field(
            default=Template(
                "Fix the previous solution based on the following information.\n\n"
                "# PLAN:\n${plan}\n"
                "# CODE:\n${training_code}\n"
                "# ISSUES:\n${review}\n"
                "# ERRORS:\n${problems}\n"
                "# PREVIOUS ATTEMPTS, IF ANY:\n${history}\n\n"
                "Correct the code, train the model, compute and print the evaluation metric, and save the model in './working'."
            )
        )
        prompt_training_review: Template = field(
            default=Template(
                "Review the solution to enhance test performance and fix issues.\n\n"
                "# TASK: ${problem_statement}\n"
                "# PLAN: ${plan}\n"
                "# CODE: ${training_code}\n"
                "# ERRORS: ${problems}\n"
                "# PREVIOUS ATTEMPTS, IF ANY: ${history}\n\n"
                "Suggest a single, actionable improvement considering previous reviews."
            )
        )
        # prompts used in generating, fixing or reviewing prediction code
        prompt_inference_base: Template = field(default=Template("Experienced ML Engineer deploying a trained model."))
        prompt_inference_generate: Template = field(
            default=Template(
                "Write an inference script.\n\n"
                "# TASK: ${problem_statement}\n"
                "# PLAN: ${plan}\n"
                "# CODE: ${training_code}\n"
                "# PREVIOUS ATTEMPTS, IF ANY: ${context}\n\n"
                "Load 'model.joblib' from './working', preprocess inputs, and generate predictions. Save results as 'predictions.csv'. "
                "Handle errors gracefully. Use ${allowed_packages}."
            )
        )
        prompt_inference_fix: Template = field(
            default=Template(
                "Fix the inference script to resolve issues.\n\n"
                "# CODE: ${inference_code}\n"
                "# ISSUES: ${review}\n"
                "# ERRORS: ${problems}\n\n"
                "Ensure compatibility with the training pipeline and handle errors gracefully."
            )
        )
        prompt_inference_review: Template = field(
            default=Template(
                "Review the inference script for improvements.\n\n"
                "# TASK: ${problem_statement}\n"
                "# PLAN: ${plan}\n"
                "# CODE: ${inference_code}\n"
                "# PREVIOUS ATTEMPTS, IF ANY: ${context}\n\n"
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
