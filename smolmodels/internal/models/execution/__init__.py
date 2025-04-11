"""
Execution module for running generated code.
"""


def create_executor(distributed=False, **kwargs):
    """Factory function to create the appropriate executor."""
    from smolmodels.internal.models.execution.process_executor import ProcessExecutor

    if distributed:
        try:
            from smolmodels.internal.models.execution.ray_executor import RayExecutor

            return RayExecutor(**kwargs)
        except ImportError:
            # Fall back to process executor if Ray is not available
            return ProcessExecutor(**kwargs)
    else:
        return ProcessExecutor(**kwargs)
