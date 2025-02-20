"""
This module provides the DatasetAdapter class, which converts various dataset formats
into a standard Pandas DataFrame representation. This enables the library to accept multiple
dataset types as inputs, while ensuring consistency and interoperability.
"""

from typing import Any, Dict, List

import pandas as pd


class DatasetAdapter:
    """
    A utility class for converting different dataset formats into Pandas DataFrames.

    This class provides a standardized method for handling structured datasets,
    ensuring compatibility with downstream processing steps.

    Currently, the class supports:
      - Pandas DataFrames (returns a copy).

    Future extensions will include:
      - Support for lazy datasets (e.g., Generators, Iterators).
      - Integration with PyTorch, TensorFlow, and Hugging Face datasets.
    """

    @staticmethod
    def coerce(dataset: Any) -> pd.DataFrame:
        """
        Ensures datasets are of a supported type.

        :param dataset: The dataset to convert. Must be a Pandas DataFrame or NumPy array.
        :return: A Pandas DataFrame containing the dataset.
        :raises ValueError: If the dataset type is unsupported.
        """
        if isinstance(dataset, pd.DataFrame):
            return dataset
        # TODO: Add support for NumPy arrays, Torch tensors, and other types
        # TODO: Lazy datasets (Generators, Iterators)
        else:
            raise ValueError(f"Unsupported dataset type: {type(dataset)}")

    @staticmethod
    def features(datasets: Dict[str, pd.DataFrame]) -> List[str]:
        """
        Extracts the feature names from the given datasets.

        :param datasets: A dictionary of dataset names and their corresponding data.
        :return: A list of feature names.
        """
        features = []
        for name, dataset in datasets.items():
            # Tabular data: extract column names
            if isinstance(dataset, pd.DataFrame):
                features.extend(dataset.columns)
            # TODO: Add support for NumPy arrays, Torch tensors, and other types
            # Array data: treat entire array as a single feature
            # elif isinstance(dataset, (np.ndarray, torch.Tensor)):
            #     features.append(name)
            else:
                raise ValueError(f"Unsupported dataset type: {type(dataset)}")
        return features
