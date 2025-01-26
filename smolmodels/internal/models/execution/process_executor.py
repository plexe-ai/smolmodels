"""
Module: ProcessExecutor for Isolated Python Code Execution

This module provides an implementation of the `Executor` interface for executing Python code snippets
in an isolated process. It captures stdout, stderr, exceptions, and stack traces, and enforces
timeout limits on execution.

Classes:
    - RedirectQueue: A helper class to redirect stdout and stderr to a multiprocessing Queue.
    - ProcessExecutor: A class to execute Python code snippets in an isolated process.

Usage:
    Create an instance of `ProcessExecutor`, providing the Python code, working directory, and timeout.
    Call the `run` method to execute the code and return the results in an `ExecutionResult` object.

Exceptions:
    - Raises `RuntimeError` if the child process fails unexpectedly.

"""

import logging
import queue
import subprocess
import sys
import time
from pathlib import Path

from smolmodels.internal.models.execution.executor import ExecutionResult, Executor
from smolmodels.config import config

logger = logging.getLogger("plexe")


class RedirectQueue:
    """
    Redirect stdout and stderr to a multiprocessing Queue.

    This class acts as a file-like object to capture messages sent to stdout and stderr
    and redirect them to a multiprocessing queue for further processing.
    """

    def __init__(self):
        """
        Initialize the RedirectQueue.

        Args:
            queue (Queue): The queue to write messages to.
        """
        self.queue = queue

    def write(self, msg: str):
        """
        Write a message to the queue.

        Args:
            msg (str): The message to write.
        """
        self.queue.put(msg)

    def flush(self):
        """No-op method to satisfy the file-like interface."""
        pass


class ProcessExecutor(Executor):
    """
    Execute Python code snippets in an isolated process.

    The `ProcessExecutor` class implements the `Executor` interface, allowing Python code
    snippets to be executed with strict isolation, output capture, and timeout enforcement.
    """

    def __init__(
        self,
        code: str,
        working_dir: Path | str,
        timeout: int = 3600,
        code_execution_file_name: str = config.execution.runfile_name,
        dataset_path: str = None,
    ):
        """
        Initialize the ProcessExecutor.

        Args:
            code (str): The Python code to execute.
            working_dir (Path | str): The working directory for execution.
            timeout (int): The maximum allowed execution time in seconds.
            code_execution_file_name (str): The filename to use for the executed script.
        """
        super().__init__(code, timeout)
        self.working_dir = Path(working_dir).resolve()
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.execution_file_name = code_execution_file_name
        self.dataset_path = dataset_path

    def run(self) -> ExecutionResult:
        """Execute code in a subprocess and return results."""
        logger.debug("REPL is executing code")
        start_time = time.time()

        # Write code to file
        code_file = self.working_dir / self.execution_file_name
        with open(code_file, "w") as f:
            f.write(self.code)

        try:
            # Execute the code in a subprocess
            process = subprocess.Popen(
                [sys.executable, str(code_file)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(self.working_dir),
                text=True,
            )

            stdout, stderr = process.communicate(timeout=self.timeout)
            exec_time = time.time() - start_time

            # Check for model artifacts in working directory
            model_artifacts = {}
            if (self.working_dir / "model.joblib").exists():
                model_artifacts["model"] = str(self.working_dir / "model.joblib")

            if process.returncode != 0:
                return ExecutionResult(
                    term_out=[stdout],
                    exec_time=exec_time,
                    exc_type="RuntimeError",
                    exc_info={"args": [stderr]},
                    exc_stack=None,
                    model_artifacts=model_artifacts,
                )

            return ExecutionResult(
                term_out=[stdout],
                exec_time=exec_time,
                exc_type=None,
                exc_info=None,
                exc_stack=None,
                model_artifacts=model_artifacts,
            )

        except subprocess.TimeoutExpired:
            process.kill()
            return ExecutionResult(
                term_out=[],
                exec_time=self.timeout,
                exc_type="TimeoutError",
                exc_info={"args": [f"Execution exceeded {self.timeout}s timeout"]},
                exc_stack=None,
                model_artifacts={},
            )
        finally:
            try:
                code_file.unlink()
            except:
                raise

    def cleanup(self):
        """Required by abstract base class."""
        pass
