"""
This module provides a generic Registry pattern implementation for storing and retrieving objects by name.
"""

from typing import TypeVar, Generic, Dict, List

T = TypeVar("T")


class Registry(Generic[T]):
    """
    Base registry for storing and retrieving objects by name.

    This class implements the Singleton pattern so that registry instances are shared
    across the application. It provides methods for registering, retrieving, and
    managing objects in a type-safe manner.
    """

    _instance = None
    _items = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Registry, cls).__new__(cls)
            cls._items = {}
        return cls._instance

    def register(self, name: str, item: T) -> None:
        """
        Register an item with a given name.

        :param name: Unique identifier for the item
        :param item: The item to register
        """
        self._items[name] = item

    def get(self, name: str) -> T:
        """
        Retrieve an item by name.

        :param name: The name of the item to retrieve
        :return: The registered item
        :raises KeyError: If the item is not found in the registry
        """
        if name not in self._items:
            raise KeyError(f"Item '{name}' not found in registry")
        return self._items[name]

    def get_multiple(self, names: List[str]) -> Dict[str, T]:
        """
        Retrieve multiple items by name.

        :param names: List of item names to retrieve
        :return: Dictionary mapping item names to items
        :raises KeyError: If any item is not found in the registry
        """
        return {name: self.get(name) for name in names}

    def clear(self) -> None:
        """
        Clear all registered items.
        """
        self._items.clear()

    def list(self) -> List[str]:
        """
        List all registered item names.

        :return: List of item names in the registry
        """
        return list(self._items.keys())

    def __contains__(self, name: str) -> bool:
        """
        Check if an item exists in the registry.

        :param name: The name to check
        :return: True if the item exists, False otherwise
        """
        return name in self._items
