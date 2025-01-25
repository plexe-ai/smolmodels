"""
Unit tests for the ProcessExecutor class and its associated components.

These tests validate the functionality of the following:
- RedirectQueue: Ensures that stdout and stderr redirection to a queue behaves as expected.
- ProcessExecutor: Tests execution of Python code in an isolated process, including handling of:
  - Successful execution.
  - Timeouts.
  - Exceptions raised during execution.
  - Cleanup and file removal logic.

The tests use pytest as the test runner and employ mocking to isolate external dependencies.
"""

import queue
import os
from multiprocessing import Queue
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from smolmodels.internal.models.execution.executor import ExecutionResult
from smolmodels.internal.models.execution.process_executor import ProcessExecutor, RedirectQueue


class TestRedirectQueue:
    def setup_method(self):
        self.queue = Queue()
        self.redirect_queue = RedirectQueue(self.queue)

    def test_write(self):
        self.redirect_queue.write("Test Message")
        assert self.queue.get() == "Test Message"

    def test_flush(self):
        self.redirect_queue.flush()  # Should do nothing without error


class TestProcessExecutor:
    def setup_method(self):
        self.code = "print('Hello, World!')"
        self.working_dir = Path(os.getcwd())
        self.timeout = 5
        self.agent_file_name = "test_runfile.py"
        self.process_executor = ProcessExecutor(self.code, self.working_dir, self.timeout, self.agent_file_name)

    @patch("smolmodels.internal.models.execution.process_executor.Process")
    def test_create_process(self, mock_process):
        self.process_executor._create_process()
        mock_process.assert_called_once()
        assert self.process_executor.process is not None

    def test_cleanup_no_process(self):
        self.process_executor.cleanup()  # No process, should exit gracefully

    @patch("smolmodels.internal.models.execution.process_executor.Process")
    def test_cleanup_with_process(self, mock_process):
        mock_proc_instance = MagicMock()
        mock_process.return_value = mock_proc_instance
        self.process_executor._create_process()
        self.process_executor.cleanup()
        mock_proc_instance.terminate.assert_called_once()
        mock_proc_instance.close.assert_called_once()

    @patch("smolmodels.internal.models.execution.process_executor.RedirectQueue")
    def test_child_proc_setup(self, mock_redirect_queue):
        result_outq = Queue()
        self.process_executor._child_proc_setup(result_outq)
        mock_redirect_queue.assert_called_once_with(result_outq)

    @patch("smolmodels.internal.models.execution.process_executor.Process")
    @patch("smolmodels.internal.models.execution.process_executor.Queue")
    def test_run_successful_execution(self, mock_queue, mock_process):
        mock_queue.return_value.get.side_effect = [
            ("state:ready",),
            ("state:finished", None, None, None),
            "Hello, World!",
            "<|EOF|>",
        ]
        mock_proc_instance = MagicMock()
        mock_process.return_value = mock_proc_instance
        result = self.process_executor.run()
        assert isinstance(result, ExecutionResult)
        assert "Hello, World!" in result.term_out

    @patch("smolmodels.internal.models.execution.process_executor.Process")
    @patch("smolmodels.internal.models.execution.process_executor.Queue")
    def test_run_timeout(self, mock_queue, mock_process):
        mock_queue.return_value.get.side_effect = queue.Empty
        mock_proc_instance = MagicMock()
        mock_process.return_value = mock_proc_instance
        with pytest.raises(RuntimeError):
            self.process_executor.run()

    @patch("smolmodels.internal.models.execution.process_executor.Process")
    @patch("smolmodels.internal.models.execution.process_executor.Queue")
    def test_run_exception(self, mock_queue, mock_process):
        mock_queue.return_value.get.side_effect = [
            ("state:ready",),
            ("state:finished", "Exception", {"args": ["error occurred"]}, []),
            "<|EOF|>",
        ]
        mock_queue.return_value.empty.side_effect = [False, True]
        mock_proc_instance = MagicMock()
        mock_process.return_value = mock_proc_instance
        result = self.process_executor.run()
        assert isinstance(result, ExecutionResult)
        assert result.exc_type == "Exception"
        assert "error occurred" in result.exc_info["args"]

    def test_run_session_creates_and_removes_file(self):
        code_in_queue = Queue()
        result_out_queue = Queue()
        event_out_queue = Queue()
        code_in_queue.put("print('test')")

        with patch.object(self.process_executor, "_child_proc_setup", return_value=None):
            self.process_executor._run_session(code_in_queue, result_out_queue, event_out_queue)

        assert os.path.exists(self.agent_file_name) is False

    @patch("builtins.open")
    @patch("smolmodels.internal.models.execution.process_executor.os.remove")
    def test_run_session_handles_missing_file(self, mock_remove, mock_open):
        mock_remove.side_effect = FileNotFoundError
        code_inq = Queue()
        result_outq = Queue()
        event_outq = Queue()
        code_inq.put("print('test')")

        with patch.object(self.process_executor, "_child_proc_setup", return_value=None):
            self.process_executor._run_session(code_inq, result_outq, event_outq)
        mock_remove.assert_called_once_with(self.agent_file_name)


if __name__ == "__main__":
    pytest.main()
