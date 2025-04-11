"""
This module provides utilities for artifact storage and retrieval.

These utilities work with the Artifact class to provide a registry for storing
and retrieving model artifacts by name.
"""

import logging
from typing import Union, List
from pathlib import Path

from smolmodels.internal.common.registries.base import Registry
from smolmodels.internal.models.entities.artifact import Artifact

logger = logging.getLogger(__name__)


class ArtifactRegistry(Registry[Artifact]):
    """
    Registry for model artifacts.

    This class extends the base Registry to provide specialized methods for
    working with model artifacts, including registration from file paths
    and batch operations.
    """

    def register_path(self, name: str, path: Union[str, Path]) -> None:
        """
        Register an artifact from a file path.

        :param name: Name to assign to the artifact
        :param path: Path to the artifact file
        :raises ValueError: If the path does not exist
        """
        path_obj = Path(path)
        if not path_obj.exists():
            raise ValueError(f"Artifact path does not exist: {path}")

        artifact = Artifact.from_path(path_obj)
        self.register(name, artifact)
        logger.debug(f"Registered artifact '{name}' from path: {path}")

    def register_batch(self, paths: List[Union[str, Path]]) -> List[str]:
        """
        Register multiple artifacts from paths and return their registry names.

        :param paths: List of paths to register as artifacts
        :return: List of registered artifact names
        """
        names = []
        for i, path in enumerate(paths):
            try:
                if isinstance(path, (str, Path)):
                    path_obj = Path(path)
                    name = f"artifact_{i}_{path_obj.name}"
                    self.register_path(name, path_obj)
                    names.append(name)
            except Exception as e:
                logger.warning(f"Failed to register artifact from path {path}: {str(e)}")

        logger.info(f"Registered batch of {len(names)} artifacts")
        return names

    def get_artifact_paths(self, names: List[str]) -> List[Path]:
        """
        Get paths for multiple artifacts.

        :param names: List of artifact names
        :return: List of paths (only for path-based artifacts)
        """
        paths = []
        artifacts = self.get_multiple(names)

        for name, artifact in artifacts.items():
            if artifact.is_path():
                paths.append(artifact.path)
            else:
                logger.warning(f"Artifact '{name}' is not path-based")

        return paths
