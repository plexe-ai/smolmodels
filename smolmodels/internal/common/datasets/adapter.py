# smolmodels/internal/common/datasets/adapter.py

"""
This module provides the DatasetAdapter class, which converts various dataset formats
into a standard Pandas DataFrame representation. This enables the library to accept multiple
dataset types as inputs, while ensuring consistency and interoperability.
"""

from typing import Any
import pandas as pd
import numpy as np


class DatasetAdapter:
    """
    A utility class for converting different dataset formats into Pandas DataFrames.

    This class provides a standardized method for handling structured datasets,
    ensuring compatibility with downstream processing steps.

    Currently, the class supports:
      - Pandas DataFrames (returns a copy).
      - NumPy arrays (converted to a DataFrame).

    Future extensions will include:
      - Support for lazy datasets (e.g., Generators, Iterators).
      - Integration with PyTorch, TensorFlow, and Hugging Face datasets.
    """

    @staticmethod
    def convert(dataset: Any) -> pd.DataFrame:
        """
        Convert a dataset into a Pandas DataFrame.

        If the dataset is already a Pandas DataFrame, a copy is returned.
        If the dataset is a NumPy array, it is converted into a DataFrame.

        :param dataset: The dataset to convert. Must be a Pandas DataFrame or NumPy array.
        :return: A Pandas DataFrame containing the dataset.
        :raises ValueError: If the dataset type is unsupported.
        """
        if isinstance(dataset, pd.DataFrame):
            return dataset.copy()
        elif isinstance(dataset, np.ndarray):
            return pd.DataFrame(dataset)
        # TODO: Add support for lazy datasets (Generators, Iterators)
        # TODO: Add support for PyTorch, TensorFlow, and Hugging Face datasets
        else:
            raise ValueError(f"Unsupported dataset type: {type(dataset)}")
