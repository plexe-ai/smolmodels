"""Data models for MLE Bench runner."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TestResult:
    """Structured class to store test results"""

    name: str
    success: bool
    submission_path: Optional[str] = None
    model_path: Optional[str] = None
    failure_reason: Optional[str] = None


@dataclass
class SubmissionInfo:
    """Structured class to store submission information"""

    competition_id: str
    submission_path: str
