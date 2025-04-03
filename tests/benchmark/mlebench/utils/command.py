"""Command execution utilities for MLE Bench runner."""

import subprocess
import sys
from contextlib import contextmanager
import os
from pathlib import Path


class CommandRunner:
    """Class to handle command execution and error handling"""

    @staticmethod
    def run(command, error_message, success_message=None):
        """Run a shell command and handle errors"""
        try:
            subprocess.run(command, check=True, text=True)
            if success_message:
                print(f"✅ {success_message}")
        except subprocess.CalledProcessError as e:
            print(f"❌ {error_message}")
            print(f"❌ Error details: {e}")
            sys.exit(1)
        except FileNotFoundError as e:
            print(f"❌ {str(e)}")
            print(f"❌ Command not found: {' '.join(command)}")
            print(
                "❌ This usually means that the required tool is not installed or not in the PATH. "
                "Please install the required dependencies and try again."
            )
            sys.exit(1)


@contextmanager
def working_directory(path):
    """Context manager for changing the current working directory"""
    prev_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)
