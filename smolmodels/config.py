"""
Configuration for the smolmodels library.
"""

from dataclasses import dataclass, field
from string import Template
from typing import List


@dataclass(frozen=True)
class _Config:
    @dataclass(frozen=True)
    class _FileStorageConfig:
        model_cache_dir: str = field(default=".smolcache/")

    @dataclass(frozen=True)
    class _LoggingConfig:
        level: str = field(default="INFO")
        format: str = field(default="[%(asctime)s - %(name)s - %(levelname)s - (%(threadName)-10s)]: - %(message)s")

    @dataclass(frozen=True)
    class _ModelSearchConfig:
        initial_nodes: int = field(default=3)
        max_nodes: int = field(default=10)
        max_fixing_attempts_train: int = field(default=3)
        max_fixing_attempts_predict: int = field(default=10)
        max_time_elapsed: int = field(default=600)

    @dataclass(frozen=True)
    class _ExecutionConfig:
        timeout: int = field(default=300)
        runfile_name: str = field(default="execution_script.py")
        training_data_path: str = field(default="training_data.parquet")

    @dataclass(frozen=True)
    class _CodeGenerationConfig:
        allowed_packages: List[str] = field(
            default_factory=lambda: ["pandas", "numpy", "scikit-learn", "joblib", "mlxtend", "xgboost", "pyarrow"]
        )
        k_fold_validation: int = field(default=5)
        # prompts used in generating plans or making decisions
        prompt_planning_base: Template = field(
            default=Template("You are an experienced ML Engineer planning a solution to a Kaggle competition.")
        )
        prompt_planning_select_metric: Template = field(
            default=Template(
                "Select what machine learning model metric is most appropriate to optimise for this task.\n\n"
                "The task is:\n${problem_statement}\n\n"
                "Tell me the name of the metric, and whether higher or lower values are better. If the metric has a "
                "specific target value, please provide that too. Select a simple metric that is appropriate for the "
                "task, but also widely known of and used in the machine learning community."
            )
        )
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
                "Write a solution plan for the machine learning problem outlined below. The solution must produce "
                "a model that achieves the best possible performance on ${metric}.\n\n"
                "# TASK:\n${problem_statement}\n\n"
                "# PREVIOUS ATTEMPTS, IF ANY:**\n${context}\n\n"
                "The solution concept should be explained in 3-5 sentences. Do not include an implementation of the "
                "solution, though you can include small code snippets if relevant to explain the plan. "
                "Do not suggest doing EDA, ensembling, or hyperparameter tuning. "
                "The solution should be feasible using only ${allowed_packages}, and no other non-standard libraries. "
            )
        )
        prompt_schema_base: Template = field(
            default=Template("You are an expert ML engineer identifying target variables.")
        )

        prompt_schema_identify_target: Template = field(
            default=Template(
                "Given these columns from a dataset:\n"
                "${columns}\n\n"
                "For this ML task: ${intent}\n\n"
                "Which column is the target/output variable? Return ONLY the exact column name, nothing else."
            )
        )
        prompt_schema_generate_from_intent: Template = field(
            default=Template(
                "Generate appropriate input and output schemas for this machine learning task.\n\n"
                "Task description: ${intent}\n\n"
                "The ${input_schema} should contain features needed for prediction.\n"
                "The ${output_schema} should contain what needs to be predicted.\n"
                "Return your response as a valid JSON object.\n"
                'Use only these types: "int", "float", "str", "bool".'
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
                "The script must assume that the dataset is in the current working directory as a parquet file "
                "called ${training_data_path}. "
                "The script must train the model, compute and print the final evaluation metric to standard output, "
                "and save the model as 'model.joblib' in the current working directory. Use only ${allowed_packages}. "
                "Do NOT use any packages that are not part of this list of the Python standard library."
                "Do not skip steps or combine preprocessors and models in the same joblib file."
            )
        )
        prompt_training_fix: Template = field(
            default=Template(
                "Fix the previous solution based on the following information.\n\n"
                "# PLAN:\n${plan}\n"
                "# CODE:\n${training_code}\n"
                "# ISSUES:\n${review}\n"
                "# ERRORS:\n${problems}\n"
                "Correct the code, train the model, compute and print the evaluation metric, and save the model in "
                "the current working directory as 'model.joblib'. Use only ${allowed_packages}. Do NOT use any "
                "packages that are not part of this list of the Python standard library. Assume the training "
                "data is in the current working directory as a parquet file called ${training_data_path}."
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
        prompt_inference_base: Template = field(
            default=Template("You are an experienced ML Engineer deploying a trained model.")
        )
        prompt_inference_generate: Template = field(
            default=Template(
                "Complete the following Python script, such that it can be used to get predictions from a machine "
                "learning model. The script to complete is the below:\n\n"
                "```python\n"
                "# todo: any library imports you need go here"
                "\n"
                "# todo: load model binaries here at the module level\n"
                "\n"
                "def predict(sample: dict) -> dict:\n"
                "    # todo: prediction code goes here\n"
                "    pass\n"
                "```\n\n"
                "You can add any imports or functionality to the code, but you must not change the overall structure "
                "or the signature of the predict() function. The input parameter 'sample' will take a single input "
                "sample for which a model inference must be produced. The contents of 'sample' will be as per the "
                "input schema below, and the contents of the returned dictionary must be as per the output schema "
                "below.\n\n"
                "# INPUT SCHEMA:\n```python\n${input_schema}```\n\n"
                "# OUTPUT SCHEMA:\n```python\n${output_schema}```\n\n"
                "The model in question has been trained using the following ML training script. Take a look at the "
                "script in order to understand what type of model is being used, how it needs to be loaded, and what "
                "type of input it expects.\n\n"
                "# TRAINING CODE FOR REFERENCE:\n```python\n${training_code}```\n\n"
                "The script must not use any packages that are not in ${allowed_packages}. Return only the completed "
                "inference script, with no external explanations or commentary."
            )
        )
        prompt_inference_fix: Template = field(
            default=Template(
                "Fix the provided Python machine learning inference script based on the information provided. The "
                "script is expected to have the following structure:\n\n"
                "```python\n"
                "# todo: any library imports go here"
                "\n"
                "# todo: load model binaries here at the module level\n"
                "\n"
                "def predict(sample: dict) -> dict:\n"
                "    # todo: prediction code goes here\n"
                "    pass\n"
                "```\n\n"
                "# INFERENCE CODE TO FIX:\n```python\n${inference_code}```\n"
                "# IDENTIFIED ISSUES:\n${review}\n\n"
                "# SPECIFIC ERRORS:\n${problems}\n\n"
                "You must not change the signature of the predict() function, unless this was specifically called out "
                "as one of the problems in the issues above. You also must not change the locations from which files "
                "are loaded, or any of the packages being imported, unless explicitly called out as part of the fix "
                "in the issues above. Return an explanation of the fix, followed by the fixed inference script."
            )
        )
        prompt_inference_review: Template = field(
            default=Template(
                "Review the Python machine learning inference script below to fix any issues. The script should "
                "adhere to the following structure:\n\n"
                "```python\n"
                "# todo: any library imports go here"
                "\n"
                "# todo: load model binaries here at the module level\n"
                "\n"
                "def predict(sample: dict) -> dict:\n"
                "    # todo: prediction code goes here\n"
                "    pass\n"
                "```\n\n"
                "Here is the script that needs to be reviewed:\n"
                "# INFERENCE CODE TO REVIEW:\n```python\n${inference_code}```\n\n"
                "The input schema for the 'sample' parameter is as follows:\n```python\n${input_schema}```\n\n"
                "The output schema for the return value is as follows:\n```python\n${output_schema}```\n\n"
                "The model itself was generated by the following training code:\n```python\n${training_code}```\n\n"
                "When doing a static analysis of the code, the following issues were spotted:\n${problems}\n\n"
                "Suggest a single, actionable improvement for fixing the issues."
            )
        )

    @dataclass(frozen=True)
    class _DataGenerationConfig:
        pass  # todo: implement

    # configuration objects
    file_storage: _FileStorageConfig = field(default_factory=_FileStorageConfig)
    logging: _LoggingConfig = field(default_factory=_LoggingConfig)
    model_search: _ModelSearchConfig = field(default_factory=_ModelSearchConfig)
    code_generation: _CodeGenerationConfig = field(default_factory=_CodeGenerationConfig)
    execution: _ExecutionConfig = field(default_factory=_ExecutionConfig)
    data_generation: _DataGenerationConfig = field(default_factory=_DataGenerationConfig)


# Instantiate configuration
def load_config() -> _Config:
    return _Config()


config: _Config = load_config()
