"""
Configuration for the smolmodels library.
"""

import logging
import warnings
from dataclasses import dataclass, field
from string import Template
from typing import List
import sys


# configure warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


@dataclass(frozen=True)
class _Config:
    @dataclass(frozen=True)
    class _FileStorageConfig:
        model_cache_dir: str = field(default=".smolcache/")
        model_dir: str = field(default="model_files/")

    @dataclass(frozen=True)
    class _LoggingConfig:
        level: str = field(default="INFO")
        format: str = field(default="[%(asctime)s - %(name)s - %(levelname)s - (%(threadName)-10s)]: - %(message)s")

    @dataclass(frozen=True)
    class _ModelSearchConfig:
        initial_nodes: int = field(default=3)
        max_nodes: int = field(default=15)
        max_fixing_attempts_train: int = field(default=3)
        max_fixing_attempts_predict: int = field(default=10)
        max_time_elapsed: int = field(default=600)

    @dataclass(frozen=True)
    class _ExecutionConfig:
        timeout: int = field(default=300)
        runfile_name: str = field(default="execution_script.py")

    @dataclass(frozen=True)
    class _CodeGenerationConfig:
        allowed_packages: List[str] = field(
            default_factory=lambda: [
                "pandas",
                "numpy",
                "scikit-learn",
                "joblib",
                "mlxtend",
                "xgboost",
                "pyarrow",
                "torch",
                "statsmodels",
                "transformers",
                "tokenizers",
                "accelerate",
                "safetensors",
            ]
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
                "If appropriate, consider using pre-trained models under 20MB that can be fine-tuned with the provided data."
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
                "The script must assume that the data to be used for training and evaluation is in the following files "
                "relative to the current directory: ${training_data_files}\n\n"
                "The script must train the model, compute and print the final evaluation metric to standard output, "
                "and **save all model files directly in the CURRENT directory** with descriptive names. "
                "Do not create any subdirectories. Use only ${allowed_packages}. "
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
                "Correct the code, train the model, compute and print the evaluation metric, and save all model files "
                "directly in the current directory with descriptive names. "
                "Do not create any subdirectories. Use only ${allowed_packages}. Do NOT use any "
                "packages that are not part of this list of the Python standard library. Assume the training "
                "data is in the following files relative to the current directory ${training_data_files}."
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
        prompt_inference_model_loading: Template = field(
            default=Template(
                "Given this training code that saves model files:\n"
                "```python\n"
                "from pathlib import Path\n\n"
                "MODEL_DIR = Path('${filedir}')\n\n"
                "${training_code}\n"
                "```\n\n"
                "Generate Python code to load the model files from MODEL_DIR. The code should:\n"
                "1. Import required packages\n"
                "2. Load all model files (models, preprocessors, etc.) that were saved during training\n"
                "3. Initialize any required variables or configurations\n\n"
                "Only include the loading code, no prediction logic. Use MODEL_DIR as the path variable."
            )
        )
        prompt_inference_preprocessing: Template = field(
            default=Template(
                "Given this input schema and training code:\n\n"
                "Input Schema:\n"
                "```python\n"
                "${input_schema}\n"
                "```\n\n"
                "Training Code:\n"
                "```python\n"
                "${training_code}\n"
                "```\n\n"
                "Generate Python code to prepare a single input sample for prediction. The code should:\n"
                "1. Take an input matching the input schema\n"
                "2. Return the data in the format expected by the model\n\n"
                "Only include preprocessing logic, assume model loading is already done."
            )
        )
        prompt_inference_prediction: Template = field(
            default=Template(
                "Given this context:\n\n"
                "Output Schema:\n"
                "```python\n"
                "${output_schema}\n"
                "```\n\n"
                "Training Code:\n"
                "```python\n"
                "${training_code}\n"
                "```\n\n"
                "Model Loading Code:\n"
                "```python\n"
                "${model_loading_code}\n"
                "```\n\n"
                "Preprocessing Code:\n"
                "```python\n"
                "${preprocessing_code}\n"
                "```\n\n"
                "Generate Python code to make predictions. The code should:\n"
                "1. Take preprocessed input data from the preprocessing step\n"
                "2. Use the loaded model to make predictions\n"
                "3. Format the predictions according to the output schema\n"
                "4. Return an output matching the output schema\n\n"
                "Only include prediction logic, assume model loading and preprocessing are already done."
            )
        )
        prompt_inference_combine: Template = field(
            default=Template(
                "Combine these code sections into a complete inference script:\n\n"
                "Model Loading:\n"
                "```python\n"
                "${model_loading_code}\n"
                "```\n\n"
                "Preprocessing:\n"
                "```python\n"
                "${preprocessing_code}\n"
                "```\n\n"
                "Prediction:\n"
                "```python\n"
                "${prediction_code}\n"
                "```\n\n"
                "The script should:\n"
                "1. Follow the required structure with imports and MODEL_DIR at the top\n"
                "2. Have a predict() function that takes an input matching the input schema and returns an output matching the output schema\n"
                "3. Organize the code logically with model loading outside the predict function\n"
                "4. Handle any necessary error checking and validation\n\n"
                "MODEL_DIR should be set to: ${model_dir}"
            )
        )
        prompt_inference_fix: Template = field(
            default=Template(
                "Fix the provided Python machine learning inference script based on the information provided. The "
                "script is expected to have the following structure:\n\n"
                "```python\n"
                "import os\n"
                "from pathlib import Path\n"
                "\n"
                "# todo: add any additional imports you need here\n"
                "\n"
                "# Load model binaries from .smolcache directory\n"
                "MODEL_DIR = Path('${model_id}')\n"
                "\n"
                "def predict(sample: dict) -> dict:\n"
                "    # todo: prediction code goes here\n"
                "    pass\n"
                "```\n\n"
                "IMPORTANT: The error in the 'SPECIFIC ERRORS' indicate what went wrong when trying to load or use the model. "
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
                "import os\n"
                "from pathlib import Path\n"
                "\n"
                "# todo: add any additional imports you need here\n"
                "\n"
                "# Load model binaries from .smolcache directory\n"
                "MODEL_DIR = Path('${filedir}')\n"
                "\n"
                "def predict(sample: dict) -> dict:\n"
                "    # todo: prediction code goes here\n"
                "    pass\n"
                "```\n\n"
                "Here is the script that needs to be reviewed:\n"
                "# INFERENCE CODE TO REVIEW:\n```python\n${inference_code}```\n\n"
                "IMPORTANT: The error  in the 'SPECIFIC ERRORS' indicate what went wrong when trying to load or use the model. "
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


# Default logging configuration
def configure_logging(level: str | int = logging.INFO, file: str = None) -> None:
    # Configure the library's root logger
    sm_root_logger = logging.getLogger("smolmodels")
    sm_root_logger.setLevel(level)

    # Define a common formatter
    formatter = logging.Formatter(config.logging.format)

    stream_handler = logging.StreamHandler()
    # Only apply reconfigure if the stream supports it
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    stream_handler.setFormatter(formatter)
    sm_root_logger.addHandler(stream_handler)

    if file:
        file_handler = logging.FileHandler(file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        sm_root_logger.addHandler(file_handler)


configure_logging(level=config.logging.level)
