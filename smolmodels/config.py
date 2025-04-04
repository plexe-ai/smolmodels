"""
Configuration for the smolmodels library.
"""

import importlib
import logging
import warnings
from dataclasses import dataclass, field
from importlib.resources import files
from string import Template
from typing import List
from functools import cached_property
from jinja2 import Environment, FileSystemLoader
import sys
from pathlib import Path

from smolmodels import templates as template_module


TEMPLATE_DIR = files("smolmodels").joinpath("templates/prompts")


# configure warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def is_package_available(package_name: str) -> bool:
    """Check if a Python package is available/installed."""
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False


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
        runfile_name: str = field(default="execution_script.py")

    @dataclass(frozen=True)
    class _CodeGenerationConfig:
        # Base ML packages that are always available
        _base_packages: List[str] = field(
            default_factory=lambda: [
                "pandas",
                "numpy",
                "scikit-learn",
                "joblib",
                "mlxtend",
                "xgboost",
                "pyarrow",
                "statsmodels",
            ]
        )

        # Deep learning packages that are optional
        _deep_learning_packages: List[str] = field(
            default_factory=lambda: [
                "torch",
                "transformers",
                "tokenizers",
                "accelerate",
                "safetensors",
            ]
        )

        @property
        def allowed_packages(self) -> List[str]:
            """Dynamically determine which packages are available and can be used."""
            available_packages = self._base_packages.copy()

            # Check if deep learning packages are installed and add them if they are
            for package in self._deep_learning_packages:
                if is_package_available(package):
                    available_packages.append(package)

            return available_packages

        @property
        def deep_learning_available(self) -> bool:
            """Check if deep learning packages are available."""
            return any(is_package_available(pkg) for pkg in self._deep_learning_packages)

        k_fold_validation: int = field(default=5)
        # prompts used in generating plans or making decisions
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

        @property
        def prompt_planning_generate_plan(self) -> Template:
            """
            Dynamically generate the plan generation template.
            Conditionally includes fine-tuning suggestion if deep learning packages are available.
            """
            base_prompt = (
                "Write a solution plan for the machine learning problem outlined below. The solution must produce "
                "a model that achieves the best possible performance on ${metric_to_optimise}.\n\n"
            )

            # Include fine-tuning suggestion only if deep learning packages are available
            if self.deep_learning_available:
                base_prompt += "If appropriate, consider using pre-trained models under 20MB that can be fine-tuned with the provided data.\n\n"

            base_prompt += (
                "# TASK:\n${problem_statement}\n\n"
                "# PREVIOUS ATTEMPTS, IF ANY:\n${context}\n\n"
                "The solution concept should be explained in 3-5 sentences. Do not include an implementation of the "
                "solution, though you can include small code snippets if relevant to explain the plan. "
                "Do not suggest doing EDA, ensembling, or hyperparameter tuning. "
                "The solution should be feasible using only ${allowed_packages}, and no other non-standard libraries. "
            )

            return Template(base_prompt)

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


@dataclass(frozen=True)
class _CodeTemplates:
    predictor_interface: str = field(
        default=Path(importlib.import_module("smolmodels.internal.models.interfaces.predictor").__file__).read_text()
    )
    predictor_template: str = field(
        default=files(template_module).joinpath("models").joinpath("predictor.tmpl.py").read_text()
    )


@dataclass(frozen=True)
class _PromptTemplates:
    template_dir: str = field(default=TEMPLATE_DIR)

    @cached_property
    def env(self) -> Environment:
        return Environment(loader=FileSystemLoader(str(self.template_dir)))

    def _render(self, template_name: str, **kwargs) -> str:
        template = self.env.get_template(template_name)
        return template.render(**kwargs)

    def planning_system(self) -> str:
        return self._render("planning/system_prompt.jinja")

    def planning_select_metric(self, problem_statement) -> str:
        return self._render("planning/select_metric.jinja", problem_statement=problem_statement)

    def planning_generate(self, problem_statement, metric_to_optimise) -> str:
        return self._render(
            "planning/generate.jinja",
            problem_statement=problem_statement,
            metric_to_optimise=metric_to_optimise,
            allowed_packages=config.code_generation.allowed_packages,
            deep_learning_available=config.code_generation.deep_learning_available,
        )

    def training_system(self) -> str:
        return self._render("training/system_prompt.jinja")

    def training_generate(
        self, problem_statement, plan, history, allowed_packages, training_data_files, validation_data_files
    ) -> str:
        return self._render(
            "training/generate.jinja",
            problem_statement=problem_statement,
            plan=plan,
            history=history,
            allowed_packages=allowed_packages,
            training_data_files=training_data_files,
            validation_data_files=validation_data_files,
            use_validation_files=len(validation_data_files) > 0,
        )

    def training_fix(
        self, training_code, plan, review, problems, allowed_packages, training_data_files, validation_data_files
    ) -> str:
        return self._render(
            "training/fix.jinja",
            training_code=training_code,
            plan=plan,
            review=review,
            problems=problems,
            allowed_packages=allowed_packages,
            training_data_files=training_data_files,
            validation_data_files=validation_data_files,
            use_validation_files=len(validation_data_files) > 0,
        )

    def training_review(self, problem_statement, plan, training_code, problems, history, allowed_packages) -> str:
        return self._render(
            "training/review.jinja",
            problem_statement=problem_statement,
            plan=plan,
            training_code=training_code,
            problems=problems,
            history=history,
            allowed_packages=allowed_packages,
        )

    def inference_system(self) -> str:
        return self._render("inference/system_prompt.jinja")

    def inference_load(self, predictor_template, training_code) -> str:
        return self._render(
            "inference/load.jinja",
            predictor_template=predictor_template,
            training_code=training_code,
        )

    def inference_preprocess(self, inference_code, input_schema, training_code) -> str:
        return self._render(
            "inference/preprocess.jinja",
            inference_code=inference_code,
            input_schema=input_schema,
            training_code=training_code,
        )

    def inference_postprocess(self, inference_code, output_schema, training_code) -> str:
        return self._render(
            "inference/postprocess.jinja",
            inference_code=inference_code,
            output_schema=output_schema,
            training_code=training_code,
        )

    def inference_predict(self, output_schema, input_schema, training_code, inference_code) -> str:
        return self._render(
            "inference/predict.jinja",
            output_schema=output_schema,
            input_schema=input_schema,
            training_code=training_code,
            inference_code=inference_code,
        )

    def inference_combine(self, inference_code, predictor_interface_source) -> str:
        return self._render(
            "inference/combine.jinja",
            inference_code=inference_code,
            predictor_interface_source=predictor_interface_source,
        )

    def inference_fix(self, predictor_interface_source, predictor_template, inference_code, review, problems) -> str:
        return self._render(
            "inference/fix.jinja",
            predictor_interface_source=predictor_interface_source,
            predictor_template=predictor_template,
            inference_code=inference_code,
            review=review,
            problems=problems,
        )

    def inference_review(
        self,
        predictor_interface_source,
        predictor_template,
        inference_code,
        input_schema,
        output_schema,
        training_code,
        problems,
    ) -> str:
        return self._render(
            "inference/review.jinja",
            predictor_interface_source=predictor_interface_source,
            predictor_template=predictor_template,
            inference_code=inference_code,
            input_schema=input_schema,
            output_schema=output_schema,
            training_code=training_code,
            problems=problems,
        )


# Instantiate configuration and templates
config: _Config = _Config()
code_templates: _CodeTemplates = _CodeTemplates()
prompt_templates: _PromptTemplates = _PromptTemplates()


# Default logging configuration
def configure_logging(level: str | int = logging.INFO, file: str = None) -> None:
    # Configure the library's root logger
    sm_root_logger = logging.getLogger("smolmodels")
    sm_root_logger.setLevel(level)

    # Clear existing handlers to avoid duplicate logs
    sm_root_logger.handlers = []

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
