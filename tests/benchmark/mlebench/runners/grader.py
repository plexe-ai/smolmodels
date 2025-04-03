"""Grader for MLE Bench benchmark results."""

import json
from pathlib import Path
from typing import List

from mlebench.core.models import SubmissionInfo
from mlebench.utils.command import CommandRunner


class MLEBenchGrader:
    """Class to handle grading of model submissions"""

    @staticmethod
    def grade_agent(submissions: List[SubmissionInfo]):
        """Grade the agent's performance based on the test results"""
        print("📊 Grading the agent's performance...")

        # Get current working directory
        original_cwd = Path.cwd()

        # Write the list of dicts to a JSONL file
        submissions_file = original_cwd / "submissions.jsonl"
        with open(submissions_file, "w") as f:
            for submission in submissions:
                f.write(
                    json.dumps(
                        {"competition_id": submission.competition_id, "submission_path": submission.submission_path}
                    )
                    + "\n"
                )

        # Create grades directory if it doesn't exist
        grades_dir = original_cwd / "grades"
        grades_dir.mkdir(exist_ok=True)

        CommandRunner.run(
            ["mlebench", "grade", "--submission", str(submissions_file), "--output-dir", str(grades_dir)],
            "Failed to grade the agent.",
            "Agent graded successfully.",
        )
        print(f"🏆 Agent grading completed for {len(submissions)} tests.")

        return grades_dir
