"""Setup and preparation for MLE Bench runner."""

import os
import sys
import shutil

from mlebench.utils.command import CommandRunner, working_directory


class MLEBenchSetup:
    """Class to handle setup of MLE-bench framework"""

    @staticmethod
    def setup_mle_bench(config, rebuild: bool = False):
        """Set up the MLE-bench framework"""
        print("🔧 Setting up 'mle-bench' framework...")

        # First, ensure kaggle package is properly installed
        print("📦 Checking kaggle package version...")
        CommandRunner.run(
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
            CommandRunner.run(
                ["git", "clone", repo_url, repo_dir],
                f"Failed to clone '{repo_url}'.",
                f"'{repo_url}' cloned successfully into '{repo_dir}'.",
            )

        # Install MLE-bench using pip
        with working_directory(repo_dir):
            print("🔍 Skipping Git LFS setup for testing...")
            CommandRunner.run(
                ["git", "lfs", "install"], "Failed to install Git LFS.", "Git LFS installed successfully."
            )
            CommandRunner.run(
                ["git", "lfs", "fetch", "--all"],
                "Failed to fetch large files with Git LFS.",
                "Fetched all large files using Git LFS.",
            )
            CommandRunner.run(
                ["git", "lfs", "pull"],
                "Failed to pull large files with Git LFS.",
                "Pulled all large files using Git LFS.",
            )

            print("🔍 Installing 'mle-bench' and dependencies...")
            CommandRunner.run(
                [sys.executable, "-m", "pip", "install", "-e", "."],
                "Failed to install 'mle-bench'.",
                "'mle-bench' installed successfully.",
            )

    @staticmethod
    def prepare_datasets(config):
        """Prepare datasets listed in the config file"""
        print("📦 Preparing datasets for 'mle-bench'...")

        repo_dir = config.get("repo_dir")
        datasets = config.get("datasets", [])
        print(f"📂 Datasets to prepare: {datasets}")

        if not datasets:
            print("⚠️ No datasets listed in 'mle-bench-config.yaml'. Skipping dataset preparation.")
            return

        with working_directory(repo_dir):
            for dataset in datasets:
                print(f"📂 Preparing dataset: {dataset}")
                CommandRunner.run(
                    ["mlebench", "prepare", "-c", dataset, "--skip-verification"],
                    f"Failed to prepare dataset: {dataset}",
                    f"Dataset '{dataset}' prepared successfully.",
                )
