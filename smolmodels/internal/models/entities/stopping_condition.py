import time

from smolmodels.internal.models.entities.metric import Metric


class StoppingCondition:
    """
    A class to represent a stopping condition for an optimization process.
    """

    def __init__(self, max_generations: int, max_time: int, metric: Metric):
        """
        Initialize the StoppingCondition with the given parameters.

        :param max_generations: max number of solutions to try before giving up
        :param max_time: max time to spend on optimization, in seconds
        :param metric: threshold for the optimization metric, stop once this is reached
        """
        self.max_generations = max_generations
        self.max_time = max_time
        self.metric = metric

    def is_met(self, generations: int, start_time: float, metric: Metric) -> bool:
        return generations >= self.max_generations or time.time() - start_time >= self.max_time or metric >= self.metric

    def __repr__(self) -> str:
        """
        Return a string representation of the Metric object.

        :return: A string representation of the Metric.
        """
        return (
            f"StoppingCondition(max_nodes={self.max_generations!r}, max_time={self.max_time}, metric={self.metric!r})"
        )

    def __str__(self) -> str:
        """
        Return a user-friendly string representation of the Metric.

        :return: A string describing the Metric.
        """
        return (
            f"stop after trying {self.max_generations} solutions, "
            f"expending {self.max_time} seconds, or "
            f"reaching performance of at least {self.metric}"
        )
