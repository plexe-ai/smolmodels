"""
This script automates the setup, execution, and grading process for the "mle-bench" framework using smolmodels.

Usage:
    python mle_bench.py --config CONFIG_PATH --rebuild

Description:
    The script clones and sets up "mle-bench", prepares datasets, reads a configuration file
    to determine the tests to run, executes models using smolmodels, and grades their performance. The
    --rebuild flag forces the script to re-clone the "mle-bench" repository and reinstall dependencies.

Dependencies:
    - git
    - git-lfs
    - python
    - mle-bench
    - PyYAML
    - Jinja2
    - pandas
    - smolmodels

Ensure that your environment has the required permissions and Kaggle API credentials configured.
"""

import argparse
import os
import subprocess
import platformdirs
import json
import pandas as pd
import time
import smolmodels as sm

import shutil
import sys
import yaml
from jinja2 import Template, Environment, meta


def run_command(command, error_message, success_message=None):
    """Run a shell command and handle errors."""
    try:
        subprocess.run(command, check=True, text=True)
        if success_message:
            print(f"✅ {success_message}")
    except subprocess.CalledProcessError as e:
        print(f"❌ {error_message}")
        print(f"Error details: {e}")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print(f"❌ Command not found: {' '.join(command)}")
        print(
            "❌ This usually means that the required tool is not installed or not in the PATH. Please install "
            "the required dependencies and try again."
        )
        sys.exit(1)


def ensure_config_exists(rebuild: bool = False):
    """Check if `mle-bench-config.yaml` exists, and if not, generate it from `mle-bench-config.yaml.jinja`."""
    if os.path.exists("mle-bench-config.yaml") and not rebuild:
        print("✅ Configuration file 'mle-bench-config.yaml' already exists.")
        return

    # Get the script directory for finding the template
    script_dir = os.path.dirname(os.path.abspath(__file__))
    template_path = os.path.join(script_dir, "mle-bench-config.yaml.jinja")

    if not os.path.exists(template_path):
        print(f"❌ Template file '{template_path}' not found. Cannot proceed.")
        sys.exit(1)

    if rebuild:
        print(f"🔄 Rebuilding 'mle-bench-config.yaml' from '{template_path}'...")
    else:
        print(f"📝 'mle-bench-config.yaml' not found. Generating it from '{template_path}'...")

    # Load the template
    with open(template_path, "r") as template_file:
        template_content = template_file.read()

    env = Environment()
    ast = env.parse(template_content)
    template = Template(template_content)

    print(f"📝 Template loaded from {template_path}")

    # Set default values and gather user inputs for template variables
    variables = {
        "repo_dir": os.path.join(os.path.expanduser("~"), "mle-bench"),
        "provider": "openai/gpt-4o",
        "max_iterations": "3",
        "timeout": "3600",
    }

    # Allow user to override defaults
    for var in meta.find_undeclared_variables(ast):
        if not var.startswith("_"):
            if var in variables:
                prompt = f"💡 Provide a value for '{var}' (default: {variables[var]}): "
            else:
                prompt = f"💡 Provide a value for '{var}': "

            try:
                user_input = input(prompt)
                if user_input.strip():  # Only update if user provided a non-empty value
                    variables[var] = user_input
            except EOFError:
                print(f"Using default value for '{var}': {variables.get(var, '')}")

    # Render and write the config file
    config_content = template.render(**variables)

    # Filter to only include spaceship-titanic for testing
    config_yaml = yaml.safe_load(config_content)
    if "datasets" in config_yaml:
        if "spaceship-titanic" in config_yaml["datasets"]:
            config_yaml["datasets"] = ["spaceship-titanic"]
        else:
            print("⚠️ 'spaceship-titanic' not found in datasets. Using first dataset for testing.")
            config_yaml["datasets"] = [config_yaml["datasets"][0]] if config_yaml["datasets"] else []

    # Add smolmodels configurations to the config
    config_yaml["provider"] = variables["provider"]
    config_yaml["max_iterations"] = int(variables["max_iterations"])
    config_yaml["timeout"] = int(variables["timeout"])

    # Write the updated config
    with open("mle-bench-config.yaml", "w") as config_file:
        yaml.dump(config_yaml, config_file, default_flow_style=False)

    print("✅ 'mle-bench-config.yaml' generated successfully.")


def ensure_kaggle_credentials():
    """Ensure that Kaggle API credentials are set up."""
    print("🔑 Checking Kaggle API credentials...")
    if not os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json")):
        print(
            "❌ Kaggle API credentials not found. Please save 'kaggle.json' to '~/.kaggle/' following "
            "the instructions at https://www.kaggle.com/docs/api."
        )
        sys.exit(1)
    print("✅ Kaggle API credentials found.")


def setup_mle_bench(config, rebuild: bool = False):
    """Set up the MLE-bench framework."""
    print("🔧 Setting up 'mle-bench' framework...")

    # First, ensure kaggle package is properly installed
    print("📦 Checking kaggle package version...")
    run_command(
        [sys.executable, "-m", "pip", "show", "kaggle"],
        "Failed to check kaggle package version.",
        "Kaggle package version checked successfully.",
    )

    repo_dir = config.get("repo_dir")
    repo_url = config.get("repo_url")

    if os.path.exists(repo_dir) and not rebuild:
        print(f"📂 '{repo_dir}' repository already exists. Skipping setup step.")
        return
    else:
        if rebuild:
            print("🔄 Rebuilding 'mle-bench' repository...")
            if os.path.exists(repo_dir):
                if os.access(repo_dir, os.W_OK):
                    print(f"Removing '{repo_dir}'...")
                    shutil.rmtree(repo_dir)
                    print(f"Removed '{repo_dir}' successfully.")
                else:
                    print(f"⚠️ No write permission for '{repo_dir}'. Attempting to change permissions...")
                    os.chmod(repo_dir, 0o700)  # Grant read, write, and execute permissions to the owner
                    if os.access(repo_dir, os.W_OK):
                        print(f"Permissions changed. Removing '{repo_dir}'...")
                        shutil.rmtree(repo_dir)
                        print(f"Removed '{repo_dir}' successfully.")
                    else:
                        print(f"❌ Failed to change permissions for '{repo_dir}'. Cannot remove the directory.")
                        sys.exit(1)
            else:
                print(f"Directory '{repo_dir}' not found. Skipping removal.")
        print(f"🔍 Cloning '{repo_url}' into '{repo_dir}'...")
        run_command(
            ["git", "clone", repo_url, repo_dir],
            f"Failed to clone '{repo_url}'.",
            f"'{repo_url}' cloned successfully into '{repo_dir}'.",
        )

    os.chdir(repo_dir)

    print("🔍 Setting up Git LFS...")
    run_command(["git", "lfs", "install"], "Failed to install Git LFS.", "Git LFS installed successfully.")

    run_command(
        ["git", "lfs", "fetch", "--all"],
        "Failed to fetch large files with Git LFS.",
        "Fetched all large files using Git LFS.",
    )

    run_command(
        ["git", "lfs", "pull"], "Failed to pull large files with Git LFS.", "Pulled all large files using Git LFS."
    )

    print("🔍 Installing 'mle-bench' and dependencies...")
    run_command(
        [sys.executable, "-m", "pip", "install", "-e", "."],
        "Failed to install 'mle-bench'.",
        "'mle-bench' installed successfully.",
    )


def prepare_datasets(config):
    """Prepare datasets listed in the config file."""
    print("📦 Preparing datasets for 'mle-bench'...")

    datasets = config.get("datasets", [])
    print(f"📂 Datasets to prepare: {datasets}")
    if not datasets:
        print("⚠️ No datasets listed in 'mle-bench-config.yaml'. Skipping dataset preparation.")
        return

    for dataset in datasets:
        print(f"📂 Preparing dataset: {dataset}")
        run_command(
            ["mlebench", "prepare", "-c", dataset, "--skip-verification"],
            f"Failed to prepare dataset: {dataset}",
            f"Dataset '{dataset}' prepared successfully.",
        )


def load_config(config_path):
    """Load the test configuration from a JSON file."""
    print(f"📂 Loading test configuration from {config_path}...")
    if not os.path.exists(config_path):
        print(f"❌ Config file not found at: {config_path}")
        sys.exit(1)
    try:
        with open(config_path, "r") as config_file:
            config = yaml.safe_load(config_file)
        print("✅ Configuration loaded successfully.")
        return config
    except yaml.YAMLError as e:
        print(f"❌ Error parsing config file: {e}")
        sys.exit(1)


def run_tests(config):
    """Run tests from the configuration file using smolmodels."""
    print("🏁 Starting test execution with smolmodels...")
    submissions = []

    # Create a working directory for model outputs
    workdir = os.path.join(os.getcwd(), "workdir")
    os.makedirs(workdir, exist_ok=True)

    # Get LLM provider from config or use default
    provider = config.get("provider", "openai/gpt-4o")
    max_iterations = config.get("max_iterations", 3)
    timeout = config.get("timeout", 3600)  # Default 1 hour timeout

    print(f"🔧 Using provider: {provider}, max_iterations: {max_iterations}, timeout: {timeout}s")

    mle_bench_data_dir = f"{platformdirs.user_cache_dir()}/mle-bench/data".replace("\\", "/")
    for test_name in config.get("datasets"):
        try:
            print(f"🔍 Running test: {test_name}")
            data_dir = f"{mle_bench_data_dir}/{test_name}/prepared/public/"

            # Check if all necessary files exist
            train_file = os.path.join(data_dir, "train.csv")
            test_file = os.path.join(data_dir, "test.csv")
            description_file = os.path.join(data_dir, "description.md")
            sample_submission_file = os.path.join(data_dir, "sample_submission.csv")

            missing_files = False
            for file_path in [train_file, test_file, description_file, sample_submission_file]:
                if not os.path.exists(file_path):
                    print(f"❌ Required file not found: {file_path}")
                    missing_files = True

            if missing_files:
                print(f"❌ Skipping test {test_name} due to missing required files")
                continue

            # Read task description
            with open(description_file, "r") as f:
                task_description = f.read()

            # Create output directory for this test
            output_dir = os.path.join(workdir, test_name)
            os.makedirs(output_dir, exist_ok=True)

            # Load datasets
            print(f"📊 Loading datasets for {test_name}...")
            train_data = pd.read_csv(train_file)
            test_data = pd.read_csv(test_file)
            sample_submission = pd.read_csv(sample_submission_file)

            # Determine target column from sample submission
            target_columns = list(sample_submission.columns)
            if "id" in target_columns or "ID" in target_columns or "Id" in target_columns:
                target_columns.remove(next(col for col in target_columns if col.lower() == "id"))

            print(f"🎯 Target columns: {target_columns}")

            # Create smolmodels model
            print(f"🤖 Creating model for {test_name}...")
            model = sm.Model(
                intent=f"Solve the Kaggle competition: {test_name}. {task_description}",
            )

            # Build the model
            print(f"🏗️ Building model for {test_name}...")
            start_time = time.time()
            try:
                model.build(datasets=[train_data], provider=provider, max_iterations=max_iterations, timeout=timeout)
                build_time = time.time() - start_time
                print(f"✅ Model built successfully in {build_time:.2f} seconds")
            except Exception as e:
                print(f"❌ Failed to build model: {e}")
                continue

            # Generate predictions row by row
            print(f"🔮 Generating predictions for {test_name}...")

            # Create a copy of the sample submission to fill in
            submission_path = os.path.join(output_dir, "submission.csv")

            try:
                # Find ID column in test data
                id_col = next((col for col in test_data.columns if col.lower() == "id"), None)

                # Process each row in test data
                prediction_results = []

                print(f"📊 Processing {len(test_data)} test records...")
                for idx, row in test_data.iterrows():
                    try:
                        # Convert row to dictionary
                        row_dict = row.to_dict()

                        # Make prediction for this row
                        row_prediction = model.predict(row_dict)

                        # Store prediction with ID
                        if id_col:
                            row_prediction[id_col] = row_dict[id_col]

                        prediction_results.append(row_prediction)

                        # Log progress periodically
                        if (idx + 1) % 10 == 0 or idx == 0 or idx == len(test_data) - 1:
                            print(f"   ⏳ Processed {idx + 1}/{len(test_data)} records...")

                    except Exception as e:
                        print(f"⚠️ Error predicting row {idx}: {e}")
                        # Add empty prediction to maintain row count
                        empty_prediction = {col: None for col in target_columns}
                        if id_col:
                            empty_prediction[id_col] = row_dict[id_col]
                        prediction_results.append(empty_prediction)

                # Create a DataFrame from all the prediction results
                all_predictions_df = pd.DataFrame(prediction_results)

                # If we have ID column, make sure it's used as the join key
                if id_col and id_col in all_predictions_df.columns:
                    # Make sure we have all the expected columns from sample_submission
                    for col in sample_submission.columns:
                        if col != id_col and col not in all_predictions_df.columns:
                            # Add missing columns with None values
                            all_predictions_df[col] = None

                    # Make sure order matches sample_submission
                    all_predictions_df = (
                        all_predictions_df.set_index(id_col).reindex(index=sample_submission[id_col]).reset_index()
                    )

                # Save the prediction results
                all_predictions_df.to_csv(submission_path, index=False)
                print(f"✅ Predictions generated and submission file created at {submission_path}")
            except Exception as e:
                print(f"❌ Failed to create submission file: {e}")
                continue

            # Save model for future reference
            model_save_path = os.path.join(output_dir, f"{test_name}_model.tar.gz")
            try:
                sm.save_model(model, model_save_path)
                print(f"✅ Model saved to {model_save_path}")
            except Exception as e:
                print(f"⚠️ Failed to save model (non-critical): {e}")

            # Add to submissions list
            submissions.append({"competition_id": test_name, "submission_path": submission_path})

        except Exception as e:
            print(f"❌ Error running test {test_name}: {e}")
            continue

    return submissions


def grade_agent(submissions: list):
    """Grade the agent's performance based on the test results."""
    print("📊 Grading the agent's performance...")
    # write the list of dicts to a JSONL file
    with open("submissions.jsonl", "w") as f:
        for submission in submissions:
            f.write(json.dumps(submission) + "\n")

    run_command(
        ["mlebench", "grade", "--submission", "submissions.jsonl", "--output-dir", "./grades"],
        "Failed to grade the agent.",
        "Agent graded successfully.",
    )
    print(f"🏆 Agent grading completed for {len(submissions)} tests.")


def main(cli_args):
    print("🚀 Starting the MLE-bench Runner with SmolModels...")

    # Create workdir if it doesn't exist
    workdir = os.path.join(os.getcwd(), "workdir")
    os.makedirs(workdir, exist_ok=True)
    print(f"📁 Using working directory: {workdir}")

    # Ensure that the configuration file exists, then load it
    ensure_config_exists(cli_args.rebuild)
    config = load_config(cli_args.config)

    ensure_kaggle_credentials()

    # Ensure that the MLE-bench framework is set up and datasets are prepared
    setup_mle_bench(config, cli_args.rebuild)
    prepare_datasets(config)

    # Run tests and grade the agent
    submissions = run_tests(config)

    if submissions:
        grade_agent(submissions)
        print(f"📊 Benchmark results saved to: {os.path.join(os.getcwd(), 'grades')}")
    else:
        print("❌ No submissions were generated. Cannot grade the agent.")

    print("✅ Script completed. Thank you for using the MLE-bench Runner!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run and grade an agent on the MLE-bench framework.")
    parser.add_argument(
        "--config", type=str, required=False, default="mle-bench-config.yaml", help="Path to the configuration file."
    )
    parser.add_argument(
        "--rebuild", action="store_true", help="Force re-clone the MLE-bench repository and reinstall dependencies."
    )
    args = parser.parse_args()
    main(args)
