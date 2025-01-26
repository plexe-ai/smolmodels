from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class ExecutionResult:
    """
    Result of executing code in an environment.

    Attributes:
        term_out (list[str]): The terminal output from the execution.
        exec_time (float): The time taken to execute the code.
        exc_type (Optional[str]): The type of exception raised, if any.
        exc_info (Optional[dict]): Additional information about the exception.
        exc_stack (Optional[list[tuple]]): The stack trace of the exception.
    """

    term_out: list[str]
    exec_time: float
    exc_type: Optional[str] = None
    exc_info: Optional[dict] = None
    exc_stack: Optional[list[tuple]] = None
    model_artifacts: Optional[dict] = None
    analysis: Optional[str] = None
    performance: Optional[float] = None


class Executor(ABC):
    """
    Abstract base class for code execution environments.
    """

    @abstractmethod
    def __init__(self, code: str, timeout: int = 3600, **kwargs: Any) -> None:
        """
        Initialise the executor.

        :param: [str] code: The code to execute.
        :param: [int] timeout: Maximum execution time in seconds. Defaults to 3600.
        :param: [Any] kwargs: Additional parameters specific to the implementation.
        """
        self.code = code
        self.timeout = timeout

    @abstractmethod
    def run(self) -> ExecutionResult:
        """
        Execute the code in the defined environment.

        :return: [ExecutionResult] The results of execution, including output and errors.
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """
        Perform any necessary cleanup (e.g., terminate processes, remove temporary files).
        """
        pass
