"""
Registry for storing and retrieving callbacks during model generation process.

This module provides a registry for callbacks that can be accessed by different
components of the system, especially tools that need to notify callbacks about
events during model generation.
"""

from typing import List

from smolmodels.callbacks import Callback
from smolmodels.internal.common.registries.base import Registry


class CallbackRegistry(Registry[Callback]):
    """
    Registry for callbacks used during model generation.

    This class implements a singleton registry for storing and retrieving
    callbacks that should be notified about events during model generation.
    It is used to pass callbacks between the model generator and its tools.
    """

    def register_batch(self, callbacks: List[Callback]) -> None:
        """
        Register multiple callbacks at once.

        Args:
            callbacks: List of callback objects to register
        """
        for i, callback in enumerate(callbacks):
            # Register each callback with an index-based name
            self.register(f"callback_{i}", callback)

    def get_all(self) -> List[Callback]:
        """
        Get all registered callbacks.

        Returns:
            List of all registered callback objects
        """
        return list(self._items.values())
