"""Error handling utilities for MLE Bench runner."""

import sys
import traceback


class ErrorHandler:
    """Class to handle and format errors"""

    @staticmethod
    def handle_error(operation, context, error, exit_on_failure=False):
        """Handle exceptions with consistent formatting"""
        print(f"❌ Error during {operation}: {error}")
        if context:
            print(f"❌ Context: {context}")
        print(traceback.format_exc())
        if exit_on_failure:
            sys.exit(1)
