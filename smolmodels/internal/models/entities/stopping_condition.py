import time

from smolmodels.internal.models.entities.metric import Metric


# todo: write this out properly
class StoppingCondition:
    def __init__(self, max_generations: int, max_time: int, metric: Metric):
        self.max_generations = max_generations
        self.max_time = max_time
        self.metric = metric

    def is_met(self, generations: int, start_time: float, metric: Metric) -> bool:
        return generations >= self.max_generations or time.time() - start_time >= self.max_time or metric >= self.metric
