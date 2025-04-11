"""
This module provides a registry for dataset storage and retrieval.

This registry works with any class implementing the TabularConvertible interface
and provides a way to store and retrieve dataset objects by name across different
components of the application.
"""

import logging

from smolmodels.internal.common.datasets.interface import TabularConvertible
from smolmodels.internal.common.registries.base import Registry

logger = logging.getLogger(__name__)


class DatasetRegistry(Registry[TabularConvertible]):
    """
    Registry for storing dataset objects by name.

    This registry provides a way to access datasets across different components
    using string references, allowing tools to receive dataset names but work
    with the actual dataset objects.
    """

    # The base Registry class handles all the implementation
