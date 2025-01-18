"""
Module: Interpreter for Isolated Python Code Execution

This module provides an implementation of the `Executor` interface for executing Python code snippets
in an isolated process. It captures stdout, stderr, exceptions, and stack traces, and enforces
timeout limits on execution.

Classes:
    - RedirectQueue: A helper class to redirect stdout and stderr to a multiprocessing Queue.
    - Interpreter: A class to execute Python code snippets in an isolated process.

Usage:
    Create an instance of `Interpreter`, providing the Python code, working directory, and timeout.
    Call the `run` method to execute the code and return the results in an `ExecutionResult` object.

Exceptions:
    - Raises `RuntimeError` if the child process fails unexpectedly.

"""

import logging
import os
import queue
import sys
import time
import traceback
from multiprocessing import Process, Queue
from pathlib import Path

from smolmodels.internal.model_generation.core.execution.executor import ExecutionResult, Executor

logger = logging.getLogger("plexe")


class RedirectQueue:
    """
    Redirect stdout and stderr to a multiprocessing Queue.

    This class acts as a file-like object to capture messages sent to stdout and stderr
    and redirect them to a multiprocessing queue for further processing.
    """

    def __init__(self, queue: Queue):
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

    The `Interpreter` class implements the `Executor` interface, allowing Python code
    snippets to be executed with strict isolation, output capture, and timeout enforcement.
    """

    def __init__(
        self,
        code: str,
        working_dir: Path | str,
        timeout: int = 3600,
        agent_file_name: str = "runfile.py",
    ):
        """
        Initialize the Interpreter.

        Args:
            code (str): The Python code to execute.
            working_dir (Path | str): The working directory for execution.
            timeout (int): The maximum allowed execution time in seconds.
            agent_file_name (str): The filename to use for the executed script.
        """
        super().__init__(code, timeout)
        self.working_dir = Path(working_dir).resolve()
        assert self.working_dir.exists(), f"Working directory {self.working_dir} does not exist"
        self.agent_file_name = agent_file_name
        self.process: Process = None  # type: ignore

    def run(self) -> ExecutionResult:
        """
        Execute the provided Python code in an isolated process and return the results.

        Returns:
            ExecutionResult: The results of the execution, including output and exception details.

        Raises:
            RuntimeError: If the child process dies unexpectedly.
        """
        logger.debug("REPL is executing code")

        if self.process is not None:
            self.cleanup()
        self._create_process()

        assert self.process.is_alive()

        self.code_inq.put(self.code)
        self._wait_for_ready_state()

        start_time = time.time()

        while True:
            try:
                state = self.event_outq.get(timeout=1)
                assert state[0] == "state:finished", state
                exec_time = time.time() - start_time
                break
            except queue.Empty:
                if not self.process.is_alive():
                    msg = "REPL child process died unexpectedly"
                    logger.critical(msg)
                    while not self.result_outq.empty():
                        logger.error(f"REPL output queue dump: {self.result_outq.get()}")
                    raise RuntimeError(msg) from None
                running_time = time.time() - start_time
                if running_time > self.timeout:
                    logger.warning("Execution exceeded the time limit. Terminating the process.")
                    self.process.kill()
                    state = (None, "TimeoutError", {}, [])
                    exec_time = self.timeout
                    break

        output = self._read_output()
        e_cls_name, exc_info, exc_stack = state[1:]
        return ExecutionResult(output, exec_time, e_cls_name, exc_info, exc_stack)

    def cleanup(self):
        """
        Terminate the child process and perform cleanup.

        This method ensures that the child process is terminated, even if it fails
        to exit gracefully. It also releases resources associated with the process.
        """
        if self.process is None:
            return
        self.process.terminate()
        self.process.join(timeout=2)
        if self.process.exitcode is None:
            logger.warning("Child process failed to terminate gracefully, killing it..")
            self.process.kill()
            self.process.join()
        self.process.close()
        self.process = None

    def _child_proc_setup(self, result_outq: Queue) -> None:
        """
        Set up the child process environment.

        Args:
            result_outq (Queue): The queue to send output messages to.
        """
        os.chdir(str(self.working_dir))

        sys.path.append(str(self.working_dir))

        sys.stdout = sys.stderr = RedirectQueue(result_outq)

    def _run_session(self, code_inq: Queue, result_outq: Queue, event_outq: Queue) -> None:
        """
        Run the execution session in the child process.

        Args:
            code_inq (Queue): Queue to receive the code to execute.
            result_outq (Queue): Queue to send stdout and stderr messages to.
            event_outq (Queue): Queue to communicate execution state events.
        """
        self._child_proc_setup(result_outq)
        while True:
            # Reset the global scope for each execution to prevent state leakage.
            global_scope: dict = {}

            code = code_inq.get()
            os.chdir(str(self.working_dir))
            with open(self.agent_file_name, "w") as f:
                f.write(code)

            event_outq.put(("state:ready",))
            try:
                exec(compile(code, self.agent_file_name, "exec"), global_scope)
                event_outq.put(("state:finished", None, None, None))
            except BaseException as e:
                tb_str = "".join(traceback.format_exception(type(e), e, e.__traceback__))
                e_cls_name = type(e).__name__
                exc_info = {"args": list(map(str, e.args))} if hasattr(e, "args") else {}
                exc_stack = traceback.extract_tb(e.__traceback__)
                exc_stack = [(t.filename, t.lineno, t.name, t.line) for t in exc_stack]

                result_outq.put(tb_str)
                event_outq.put(("state:finished", e_cls_name, exc_info, exc_stack))

            try:
                os.remove(self.agent_file_name)
            except FileNotFoundError:
                pass

            result_outq.put("<|EOF|>")

    def _create_process(self) -> None:
        """
        Create the child process to execute code.

        This method initializes the multiprocessing queues and starts the child process
        that runs the code execution session.
        """
        self.code_inq, self.result_outq, self.event_outq = Queue(), Queue(), Queue()
        self.process = Process(
            target=self._run_session,
            args=(self.code_inq, self.result_outq, self.event_outq),
        )
        self.process.start()

    def _wait_for_ready_state(self):
        """
        Wait for the child process to signal readiness.

        Raises:
            RuntimeError: If the child process fails to start.
        """
        try:
            state = self.event_outq.get(timeout=10)
        except queue.Empty:
            msg = "REPL child process failed to start execution"
            logger.critical(msg)
            while not self.result_outq.empty():
                logger.error(f"REPL output queue dump: {self.result_outq.get()}")
            raise RuntimeError(msg) from None
        assert state[0] == "state:ready", state

    def _read_output(self) -> list[str]:
        """
        Read all output from the child process.

        Returns:
            list[str]: A list of output lines.
        """
        output: list[str] = []
        while not self.result_outq.empty() or not output or output[-1] != "<|EOF|>":
            output.append(self.result_outq.get())
        if output and output[-1] == "<|EOF|>":
            output.pop()
        return output
